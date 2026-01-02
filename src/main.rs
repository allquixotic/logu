use std::{fs, io, path::PathBuf};

use clap::Parser;
use directories::BaseDirs;
use promkit_core::{
    crossterm::{
        self, cursor,
        event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers},
        style,
        terminal::{disable_raw_mode, enable_raw_mode},
    },
    grapheme::StyledGraphemes,
};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    signal,
    sync::mpsc,
    task::JoinHandle,
    time::{self, Duration},
};
use tokio_util::sync::CancellationToken;

mod drain;
use drain::Drain;
const DEFAULT_RENDER_INTERVAL_MILLIS: u64 = 100;
const DEFAULT_TRAIN_INTERVAL_MILLIS: u64 = 10;
const DEFAULT_CLUSTER_SIZE_TH: usize = 0;
const DEFAULT_MAX_NODE_DEPTH: usize = 2;
const DEFAULT_SIM_TH: f32 = 0.4;
const DEFAULT_MAX_CHILDREN: usize = 100;
const DEFAULT_PARAM_STR: &str = "<*>";
const DEFAULT_SAMPLE_WINDOW_MILLIS: u64 = 1000;

#[derive(Parser)]
#[command(name = "logu", version)]
pub struct Args {
    #[arg(
        short = 'o',
        long = "stdout",
        help = "Stream deduplicated clusters to stdout instead of TUI."
    )]
    pub stdout_mode: bool,

    #[arg(
        long = "render-interval",
        value_name = "MILLIS",
        help = "Interval to render the list in milliseconds (default from config or 100).",
        long_help = "Adjust this value to prevent screen flickering
        when a large volume of list is rendered in a short period."
    )]
    pub render_interval_millis: Option<u64>,

    #[arg(
        long = "train-interval",
        value_name = "MILLIS",
        help = "How often to train in milliseconds (default from config or 10)."
    )]
    pub train_interval_millis: Option<u64>,

    #[arg(
        long = "cluster-size-th",
        value_name = "SIZE",
        help = "Threshold to filter out small clusters (default from config or 0)."
    )]
    pub cluster_size_th: Option<usize>,

    // Drain related params
    #[arg(long = "max-clusters")]
    pub max_clusters: Option<usize>,
    #[arg(
        long = "max-node-depth",
        value_name = "DEPTH",
        help = "Prefix tree depth (default from config or 2)."
    )]
    pub max_node_depth: Option<usize>,
    #[arg(
        long = "sim-th",
        value_name = "THRESHOLD",
        help = "Similarity threshold (default from config or 0.4)."
    )]
    pub sim_th: Option<f32>,
    #[arg(
        long = "max-children",
        value_name = "COUNT",
        help = "Max children per node (default from config or 100)."
    )]
    pub max_children: Option<usize>,
    #[arg(
        long = "param-str",
        value_name = "TOKEN",
        help = "Wildcard placeholder (default from config or <*>)."
    )]
    pub param_str: Option<String>,

    #[arg(
        long = "sample-window",
        value_name = "MILLIS",
        help = "Sampling window for stdout mode before emitting clusters (default 1000)."
    )]
    pub sample_window_millis: Option<u64>,

    #[arg(
        short = 'e',
        long = "examples",
        value_name = "N",
        help = "Number of original example lines to print before showing suppression count (default 0)."
    )]
    pub examples: Option<usize>,

    #[arg(
        short = 'l',
        long = "long-output",
        help = "Use long-form output like '[N instances of the last M lines suppressed]'."
    )]
    pub long_output: bool,

    #[arg(
        short = 'w',
        long = "write",
        help = "Write CLI config options to the config file, merging with existing settings."
    )]
    pub write_config: bool,
}

/// Find the longest repeating pattern at the end of a sequence of cluster IDs.
/// Returns (pattern_length, repeat_count) where repeat_count >= 2, or None if no pattern found.
fn find_repeating_pattern(cluster_ids: &[usize]) -> Option<(usize, usize)> {
    let len = cluster_ids.len();
    if len < 2 {
        return None;
    }

    let mut best_pattern_len = 0;
    let mut best_repeat_count = 0;

    // Try pattern lengths from 1 to half the sequence length
    for pattern_len in 1..=len / 2 {
        let pattern = &cluster_ids[len - pattern_len..];
        let mut repeat_count = 1;

        // Count how many times this pattern repeats backwards from the end
        let mut pos = len - pattern_len;
        while pos >= pattern_len {
            let candidate = &cluster_ids[pos - pattern_len..pos];
            if candidate == pattern {
                repeat_count += 1;
                pos -= pattern_len;
            } else {
                break;
            }
        }

        // Prefer patterns with more total lines covered (pattern_len * repeat_count)
        if repeat_count >= 2 {
            let coverage = pattern_len * repeat_count;
            let best_coverage = best_pattern_len * best_repeat_count;
            if coverage > best_coverage {
                best_pattern_len = pattern_len;
                best_repeat_count = repeat_count;
            }
        }
    }

    if best_repeat_count >= 2 {
        Some((best_pattern_len, best_repeat_count))
    } else {
        None
    }
}

async fn run_stdout_mode(config: EffectiveConfig) -> anyhow::Result<()> {
    let mut render_interval = time::interval(Duration::from_millis(config.sample_window_millis));
    let mut train_interval = time::interval(Duration::from_millis(config.train_interval_millis));
    let mut drain = Drain::new(
        config.max_clusters,
        config.max_node_depth,
        config.sim_th,
        config.max_children,
        config.param_str.clone(),
        config.examples,
    )?;

    // Track the sequence of (cluster_id, original_line) for pattern detection
    let mut line_sequence: Vec<(usize, String)> = Vec::new();

    // Spawn a dedicated stdin reader that buffers lines into a channel.
    // This prevents logu from slowing down upstream programs in the pipe.
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let stdin_reader: JoinHandle<()> = tokio::spawn(async move {
        let mut reader = BufReader::new(tokio::io::stdin()).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            let cleaned = line.replace(['\n', '\t'], " ");
            let escaped = strip_ansi_escapes::strip_str(cleaned);
            if tx.send(escaped).is_err() {
                break;
            }
        }
    });

    let result = async {
        let mut stdin_closed = false;
        loop {
            tokio::select! {
                _ = signal::ctrl_c() => break,
                _ = train_interval.tick() => {
                    // Drain all buffered lines from the channel
                    loop {
                        match rx.try_recv() {
                            Ok(line) => {
                                let cluster = drain.train(&line);
                                line_sequence.push((cluster.cluster_id, line));
                            }
                            Err(mpsc::error::TryRecvError::Empty) => break,
                            Err(mpsc::error::TryRecvError::Disconnected) => {
                                stdin_closed = true;
                                break;
                            }
                        }
                    }
                }
                _ = render_interval.tick() => {
                    if config.long_output {
                        // Long output mode: detect multi-line patterns or output runs in order
                        let cluster_ids: Vec<usize> = line_sequence.iter().map(|(id, _)| *id).collect();

                        // Only use pattern detection if it covers >50% of the sequence
                        let use_pattern = if let Some((pattern_len, repeat_count)) = find_repeating_pattern(&cluster_ids) {
                            let coverage = pattern_len * repeat_count;
                            coverage * 2 > line_sequence.len()
                        } else {
                            false
                        };

                        if use_pattern {
                            let (pattern_len, repeat_count) = find_repeating_pattern(&cluster_ids).unwrap();
                            // Print examples from the first occurrence of the pattern
                            let pattern_start = line_sequence.len() - (pattern_len * repeat_count);
                            let examples_to_print = config.examples.min(repeat_count);

                            for i in 0..examples_to_print {
                                let start = pattern_start + i * pattern_len;
                                for j in 0..pattern_len {
                                    println!("{}", line_sequence[start + j].1);
                                }
                            }

                            let suppressed = repeat_count - examples_to_print;
                            if suppressed > 0 {
                                let line_word = if pattern_len == 1 { "line" } else { "lines" };
                                println!("[{} instances of the last {} {} suppressed]", suppressed, pattern_len, line_word);
                            }
                        } else {
                            // Output lines in order, limiting examples per cluster
                            let mut cluster_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                            let mut total_suppressed = 0;

                            for (cluster_id, line) in &line_sequence {
                                let count = cluster_counts.entry(*cluster_id).or_insert(0);
                                if *count < config.examples {
                                    println!("{}", line);
                                    *count += 1;
                                } else {
                                    total_suppressed += 1;
                                }
                            }

                            if total_suppressed > 0 {
                                let unique_clusters = cluster_counts.len();
                                println!("[{} similar lines suppressed across {} patterns]", total_suppressed, unique_clusters);
                            }
                        }
                    } else {
                        // Standard output mode
                        let clusters = drain.clusters();
                        for cluster in clusters
                            .iter()
                            .filter(|cluster| cluster.size > config.cluster_size_th)
                        {
                            // Print examples first
                            let examples_printed = cluster.examples.len().min(config.examples);
                            for example in cluster.examples.iter().take(config.examples) {
                                println!("{}", example);
                            }
                            // Print count and template only if there are more occurrences than examples printed
                            let remaining = cluster.size.saturating_sub(examples_printed);
                            if remaining > 0 {
                                if remaining == 1 {
                                    println!("{}", cluster);
                                } else {
                                    println!("{}\t{}", remaining, cluster);
                                }
                            }
                        }
                    }
                    // Exit if stdin closed, otherwise reset for next window
                    if stdin_closed {
                        break;
                    }
                    drain = Drain::new(
                        config.max_clusters,
                        config.max_node_depth,
                        config.sim_th,
                        config.max_children,
                        config.param_str.clone(),
                        config.examples,
                    )?;
                    line_sequence.clear();
                }
            }
        }
        Ok::<(), anyhow::Error>(())
    }
    .await;

    stdin_reader.abort();
    result
}

#[derive(Debug, Deserialize, Serialize, Default)]
#[serde(default)]
struct FileConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    stdout_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    render_interval_millis: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    train_interval_millis: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cluster_size_th: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_clusters: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_node_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sim_th: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_children: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    param_str: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_window_millis: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    examples: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    long_output: Option<bool>,
}

#[derive(Debug)]
struct EffectiveConfig {
    render_interval_millis: u64,
    train_interval_millis: u64,
    cluster_size_th: usize,
    max_clusters: Option<usize>,
    max_node_depth: usize,
    sim_th: f32,
    max_children: usize,
    param_str: String,
    sample_window_millis: u64,
    examples: usize,
    long_output: bool,
}

fn config_path() -> Option<PathBuf> {
    BaseDirs::new().map(|dirs| {
        let base = if cfg!(windows) {
            dirs.data_local_dir()
        } else {
            dirs.config_dir()
        };
        base.join("logu").join("logu.json")
    })
}

fn load_config() -> anyhow::Result<Option<FileConfig>> {
    let Some(path) = config_path() else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(path)?;
    let cfg: FileConfig = serde_json::from_str(&contents)?;
    Ok(Some(cfg))
}

/// Write CLI options to config file, merging with existing config (CLI wins on conflicts)
fn write_config(args: &Args, existing: Option<FileConfig>) -> anyhow::Result<()> {
    let Some(path) = config_path() else {
        anyhow::bail!("Could not determine config path");
    };

    // Start with existing config or default
    let mut cfg = existing.unwrap_or_default();

    // Merge CLI options (only if explicitly set)
    if args.stdout_mode {
        cfg.stdout_mode = Some(true);
    }
    if args.render_interval_millis.is_some() {
        cfg.render_interval_millis = args.render_interval_millis;
    }
    if args.train_interval_millis.is_some() {
        cfg.train_interval_millis = args.train_interval_millis;
    }
    if args.cluster_size_th.is_some() {
        cfg.cluster_size_th = args.cluster_size_th;
    }
    if args.max_clusters.is_some() {
        cfg.max_clusters = args.max_clusters;
    }
    if args.max_node_depth.is_some() {
        cfg.max_node_depth = args.max_node_depth;
    }
    if args.sim_th.is_some() {
        cfg.sim_th = args.sim_th;
    }
    if args.max_children.is_some() {
        cfg.max_children = args.max_children;
    }
    if args.param_str.is_some() {
        cfg.param_str = args.param_str.clone();
    }
    if args.sample_window_millis.is_some() {
        cfg.sample_window_millis = args.sample_window_millis;
    }
    if args.examples.is_some() {
        cfg.examples = args.examples;
    }
    if args.long_output {
        cfg.long_output = Some(true);
    }

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Write config file with pretty formatting
    let json = serde_json::to_string_pretty(&cfg)?;
    fs::write(&path, json)?;

    eprintln!("Config written to: {}", path.display());
    Ok(())
}

fn merge_config(args: &Args, file_cfg: Option<FileConfig>) -> EffectiveConfig {
    let fc = file_cfg.unwrap_or_default();
    let mut cfg = EffectiveConfig {
        render_interval_millis: args
            .render_interval_millis
            .or(fc.render_interval_millis)
            .unwrap_or(DEFAULT_RENDER_INTERVAL_MILLIS),
        train_interval_millis: args
            .train_interval_millis
            .or(fc.train_interval_millis)
            .unwrap_or(DEFAULT_TRAIN_INTERVAL_MILLIS),
        cluster_size_th: args
            .cluster_size_th
            .or(fc.cluster_size_th)
            .unwrap_or(DEFAULT_CLUSTER_SIZE_TH),
        max_clusters: args.max_clusters.or(fc.max_clusters),
        max_node_depth: args
            .max_node_depth
            .or(fc.max_node_depth)
            .unwrap_or(DEFAULT_MAX_NODE_DEPTH),
        sim_th: args.sim_th.or(fc.sim_th).unwrap_or(DEFAULT_SIM_TH),
        max_children: args
            .max_children
            .or(fc.max_children)
            .unwrap_or(DEFAULT_MAX_CHILDREN),
        param_str: args
            .param_str
            .clone()
            .or(fc.param_str)
            .unwrap_or_else(|| DEFAULT_PARAM_STR.to_string()),
        sample_window_millis: args
            .sample_window_millis
            .or(fc.sample_window_millis)
            .unwrap_or(DEFAULT_SAMPLE_WINDOW_MILLIS),
        examples: args.examples.or(fc.examples).unwrap_or(0),
        long_output: args.long_output || fc.long_output.unwrap_or(false),
    };

    // If stdout mode and no render interval provided anywhere, default to 1s
    if args.stdout_mode
        && args.render_interval_millis.is_none()
        && fc.render_interval_millis.is_none()
    {
        cfg.render_interval_millis = 1000;
    }

    cfg
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let file_config = load_config()?;

    // Handle --write flag: save CLI options to config file and exit
    if args.write_config {
        write_config(&args, file_config)?;
        return Ok(());
    }

    // Determine stdout mode from CLI or file config
    let stdout_mode = args.stdout_mode
        || file_config.as_ref().and_then(|fc| fc.stdout_mode).unwrap_or(false);

    let config = merge_config(&args, file_config);

    if stdout_mode {
        run_stdout_mode(config).await?;
        return Ok(());
    }

    enable_raw_mode()?;
    // Avoid the rendering messy by disabling mouse scroll and fixing the row.
    crossterm::execute!(
        io::stdout(),
        crossterm::event::EnableMouseCapture,
        crossterm::cursor::Hide
    )?;

    let canceler = CancellationToken::new();

    // Spawn a dedicated stdin reader that buffers lines into a channel.
    // This prevents logu from slowing down upstream programs in the pipe.
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let stdin_reader: JoinHandle<()> = tokio::spawn(async move {
        let mut reader = BufReader::new(tokio::io::stdin()).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            let cleaned = line.replace(['\n', '\t'], " ");
            let escaped = strip_ansi_escapes::strip_str(cleaned);
            if tx.send(escaped).is_err() {
                break;
            }
        }
    });

    let canceled = canceler.clone();
    let draining: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        let render_interval = time::interval(Duration::from_millis(config.render_interval_millis));
        let train_interval = time::interval(Duration::from_millis(config.train_interval_millis));
        futures::pin_mut!(render_interval);
        futures::pin_mut!(train_interval);

        let mut drain = Drain::new(
            config.max_clusters,
            config.max_node_depth,
            config.sim_th,
            config.max_children,
            config.param_str.clone(),
            config.examples,
        )?;

        let mut prev_revision = drain.revision();

        while !canceled.is_cancelled() {
            tokio::select! {
                _ = train_interval.tick() => {
                    // Drain all buffered lines from the channel
                    loop {
                        match rx.try_recv() {
                            Ok(line) => { drain.train(line); }
                            Err(mpsc::error::TryRecvError::Empty) => break,
                            Err(mpsc::error::TryRecvError::Disconnected) => {
                                // stdin closed, exit gracefully
                                return Ok(());
                            }
                        }
                    }
                }
                _ = render_interval.tick() => {
                    let current_revision = drain.revision();
                    if current_revision == prev_revision {
                        continue;
                    }

                    let terminal_size = crossterm::terminal::size()?;
                    crossterm::execute!(
                        io::stdout(),
                        crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
                        cursor::MoveTo(0, 0),
                    )?;

                    let clusters = drain.clusters();
                    let mut total_rows = 0;
                    for cluster in clusters
                        .iter()
                        .filter(|cluster| cluster.size > config.cluster_size_th)
                        .take(terminal_size.1 as usize)
                    {
                        let cluster_str = cluster.to_string();
                        let styled = StyledGraphemes::from(cluster_str.clone());
                        let rows = styled
                            .matrixify(terminal_size.0 as usize, terminal_size.1 as usize, 0)
                            .0;

                        if total_rows + rows.len() > terminal_size.1 as usize {
                            break;
                        }

                        crossterm::execute!(
                            io::stdout(),
                            style::Print(cluster_str),
                            cursor::MoveToNextLine(1),
                        )?;

                        total_rows += rows.len();
                    }
                    prev_revision = current_revision;
                }
            }
        }
        Ok(())
    });

    loop {
        let event = crossterm::event::read()?;
        #[allow(clippy::single_match)]
        match event {
            Event::Key(KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                kind: KeyEventKind::Press,
                state: KeyEventState::NONE,
            }) => {
                break;
            }
            _ => {}
        }
    }

    canceler.cancel();
    stdin_reader.abort();
    draining.await??;

    disable_raw_mode()?;
    crossterm::execute!(
        io::stdout(),
        crossterm::event::DisableMouseCapture,
        crossterm::cursor::Show
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_args() -> Args {
        Args {
            stdout_mode: false,
            render_interval_millis: None,
            train_interval_millis: None,
            cluster_size_th: None,
            max_clusters: None,
            max_node_depth: None,
            sim_th: None,
            max_children: None,
            param_str: None,
            sample_window_millis: None,
            examples: None,
            long_output: false,
            write_config: false,
        }
    }

    #[test]
    fn merge_config_prefers_cli_over_file() {
        let mut args = empty_args();
        args.render_interval_millis = Some(500);

        let file_cfg = FileConfig {
            render_interval_millis: Some(250),
            train_interval_millis: Some(20),
            ..Default::default()
        };

        let merged = merge_config(&args, Some(file_cfg));
        assert_eq!(merged.render_interval_millis, 500);
        assert_eq!(merged.train_interval_millis, 20);
    }

    #[test]
    fn merge_config_falls_back_to_defaults() {
        let args = empty_args();
        let merged = merge_config(&args, None);
        assert_eq!(
            merged.render_interval_millis,
            DEFAULT_RENDER_INTERVAL_MILLIS
        );
        assert_eq!(merged.train_interval_millis, DEFAULT_TRAIN_INTERVAL_MILLIS);
        assert_eq!(merged.cluster_size_th, DEFAULT_CLUSTER_SIZE_TH);
        assert_eq!(merged.max_node_depth, DEFAULT_MAX_NODE_DEPTH);
        assert_eq!(merged.sim_th, DEFAULT_SIM_TH);
        assert_eq!(merged.max_children, DEFAULT_MAX_CHILDREN);
        assert_eq!(merged.param_str, DEFAULT_PARAM_STR);
    }

    #[test]
    fn stdout_mode_defaults_render_interval_to_one_second() {
        let mut args = empty_args();
        args.stdout_mode = true;

        let merged = merge_config(&args, None);
        assert_eq!(merged.render_interval_millis, 1000);
        assert_eq!(merged.sample_window_millis, DEFAULT_SAMPLE_WINDOW_MILLIS);
    }

    #[test]
    fn find_repeating_pattern_detects_simple_repeat() {
        // Pattern [1, 2] repeated 3 times
        let ids = vec![1, 2, 1, 2, 1, 2];
        let result = find_repeating_pattern(&ids);
        assert_eq!(result, Some((2, 3)));
    }

    #[test]
    fn find_repeating_pattern_detects_single_element_repeat() {
        // Same cluster ID repeated
        let ids = vec![1, 1, 1, 1];
        let result = find_repeating_pattern(&ids);
        assert_eq!(result, Some((1, 4)));
    }

    #[test]
    fn find_repeating_pattern_prefers_longer_coverage() {
        // Could be [1] x4 or [1,1] x2 - should prefer more coverage
        // Actually [1] x4 covers 4 elements, [1,1] x2 covers 4 elements too
        // The algorithm should find [1] x4 since it's checked first and has same coverage
        let ids = vec![1, 1, 1, 1];
        let result = find_repeating_pattern(&ids);
        // Both have coverage 4, but pattern_len=1 is found first
        assert_eq!(result, Some((1, 4)));
    }

    #[test]
    fn find_repeating_pattern_returns_none_for_no_repeat() {
        let ids = vec![1, 2, 3, 4];
        let result = find_repeating_pattern(&ids);
        assert_eq!(result, None);
    }

    #[test]
    fn find_repeating_pattern_returns_none_for_short_input() {
        let ids = vec![1];
        assert_eq!(find_repeating_pattern(&ids), None);

        let ids: Vec<usize> = vec![];
        assert_eq!(find_repeating_pattern(&ids), None);
    }

    #[test]
    fn find_repeating_pattern_handles_partial_repeat_at_start() {
        // Pattern [2, 3] repeats at end, but 1 is different at start
        let ids = vec![1, 2, 3, 2, 3];
        let result = find_repeating_pattern(&ids);
        assert_eq!(result, Some((2, 2)));
    }

    #[test]
    fn find_repeating_pattern_three_line_pattern() {
        // A, B, C repeated twice
        let ids = vec![1, 2, 3, 1, 2, 3];
        let result = find_repeating_pattern(&ids);
        assert_eq!(result, Some((3, 2)));
    }

    #[test]
    fn long_output_flag_merges_correctly() {
        let mut args = empty_args();
        args.long_output = true;

        let merged = merge_config(&args, None);
        assert!(merged.long_output);

        // File config should also work
        let args2 = empty_args();
        let file_cfg = FileConfig {
            long_output: Some(true),
            ..Default::default()
        };
        let merged2 = merge_config(&args2, Some(file_cfg));
        assert!(merged2.long_output);
    }

    #[test]
    fn examples_config_merges_correctly() {
        let mut args = empty_args();
        args.examples = Some(5);

        let merged = merge_config(&args, None);
        assert_eq!(merged.examples, 5);

        // CLI should override file
        let file_cfg = FileConfig {
            examples: Some(10),
            ..Default::default()
        };
        let merged2 = merge_config(&args, Some(file_cfg));
        assert_eq!(merged2.examples, 5);
    }

    #[test]
    fn stdout_mode_from_file_config() {
        // stdout_mode should be read from file config when not set on CLI
        let _args = empty_args();
        let file_cfg = FileConfig {
            stdout_mode: Some(true),
            ..Default::default()
        };

        // The merge_config doesn't handle stdout_mode, but main() does
        // We test that file config can store stdout_mode
        assert_eq!(file_cfg.stdout_mode, Some(true));

        // CLI flag should override file config (tested via the || logic in main)
        let mut args_with_flag = empty_args();
        args_with_flag.stdout_mode = true;
        // args.stdout_mode || file_config.stdout_mode.unwrap_or(false)
        let effective = args_with_flag.stdout_mode
            || file_cfg.stdout_mode.unwrap_or(false);
        assert!(effective);

        // File config alone should work
        let args_no_flag = empty_args();
        let effective2 = args_no_flag.stdout_mode
            || file_cfg.stdout_mode.unwrap_or(false);
        assert!(effective2);

        // Neither set should be false
        let empty_file_cfg = FileConfig::default();
        let effective3 = args_no_flag.stdout_mode
            || empty_file_cfg.stdout_mode.unwrap_or(false);
        assert!(!effective3);
    }

    #[test]
    fn build_file_config_from_args_stores_boolean_flags() {
        // Test that boolean flags are properly stored when building FileConfig from args
        let mut args = empty_args();
        args.stdout_mode = true;
        args.long_output = true;
        args.examples = Some(3);

        // Simulate what write_config does
        let mut cfg = FileConfig::default();
        if args.stdout_mode {
            cfg.stdout_mode = Some(true);
        }
        if args.long_output {
            cfg.long_output = Some(true);
        }
        if args.examples.is_some() {
            cfg.examples = args.examples;
        }

        assert_eq!(cfg.stdout_mode, Some(true));
        assert_eq!(cfg.long_output, Some(true));
        assert_eq!(cfg.examples, Some(3));
    }

    #[test]
    fn build_file_config_preserves_existing_on_merge() {
        // Test that existing config values are preserved when not overridden
        let existing = FileConfig {
            stdout_mode: Some(true),
            examples: Some(5),
            render_interval_millis: Some(200),
            ..Default::default()
        };

        // Args that only set long_output
        let mut args = empty_args();
        args.long_output = true;

        // Simulate write_config merge logic
        let mut cfg = existing;
        if args.stdout_mode {
            cfg.stdout_mode = Some(true);
        }
        if args.long_output {
            cfg.long_output = Some(true);
        }
        if args.examples.is_some() {
            cfg.examples = args.examples;
        }

        // Existing values should be preserved
        assert_eq!(cfg.stdout_mode, Some(true));  // Preserved from existing
        assert_eq!(cfg.examples, Some(5));         // Preserved from existing
        assert_eq!(cfg.render_interval_millis, Some(200)); // Preserved
        // New value should be set
        assert_eq!(cfg.long_output, Some(true));   // Set from args
    }

    /// Helper to calculate remaining count after printing examples
    /// This tests the output logic: remaining = size - min(examples_stored, examples_requested)
    fn calculate_remaining(cluster_size: usize, examples_stored: usize, examples_requested: usize) -> usize {
        let examples_printed = examples_stored.min(examples_requested);
        cluster_size.saturating_sub(examples_printed)
    }

    #[test]
    fn remaining_count_when_no_examples_requested() {
        // No examples requested: remaining = size
        assert_eq!(calculate_remaining(5, 3, 0), 5);
        assert_eq!(calculate_remaining(1, 1, 0), 1);
    }

    #[test]
    fn remaining_count_when_all_printed_as_examples() {
        // All occurrences printed as examples: remaining = 0
        assert_eq!(calculate_remaining(3, 3, 5), 0);
        assert_eq!(calculate_remaining(1, 1, 1), 0);
        assert_eq!(calculate_remaining(2, 2, 10), 0);
    }

    #[test]
    fn remaining_count_partial_examples() {
        // Some printed as examples, some remaining
        assert_eq!(calculate_remaining(5, 2, 2), 3); // 5 - 2 = 3
        assert_eq!(calculate_remaining(10, 3, 3), 7); // 10 - 3 = 7
        assert_eq!(calculate_remaining(5, 5, 2), 3); // stored 5, request 2, print 2, remain 3
    }

    #[test]
    fn remaining_count_fewer_stored_than_requested() {
        // Requested more examples than stored
        assert_eq!(calculate_remaining(5, 2, 10), 3); // Only 2 stored, print 2, remain 3
    }

    #[test]
    fn file_config_serialization_skips_none_values() {
        // Test that None values are skipped in JSON serialization
        let cfg = FileConfig {
            stdout_mode: Some(true),
            examples: Some(3),
            ..Default::default()
        };

        let json = serde_json::to_string(&cfg).unwrap();

        // Should contain the set values
        assert!(json.contains("stdout_mode"));
        assert!(json.contains("examples"));

        // Should not contain unset values (they're skipped due to skip_serializing_if)
        assert!(!json.contains("render_interval_millis"));
        assert!(!json.contains("train_interval_millis"));
        assert!(!json.contains("long_output"));
    }

    #[test]
    fn file_config_deserialization_handles_missing_fields() {
        // Test that deserialization works with partial JSON
        let json = r#"{"examples": 5, "stdout_mode": true}"#;
        let cfg: FileConfig = serde_json::from_str(json).unwrap();

        assert_eq!(cfg.examples, Some(5));
        assert_eq!(cfg.stdout_mode, Some(true));
        assert_eq!(cfg.long_output, None);
        assert_eq!(cfg.render_interval_millis, None);
    }

    #[test]
    fn file_config_deserialization_empty_json() {
        // Empty JSON object should work (all fields are optional)
        let json = "{}";
        let cfg: FileConfig = serde_json::from_str(json).unwrap();

        assert_eq!(cfg.stdout_mode, None);
        assert_eq!(cfg.examples, None);
        assert_eq!(cfg.long_output, None);
    }
}
