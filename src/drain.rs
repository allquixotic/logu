use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    num::NonZeroUsize,
};

use lru::LruCache;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LogCluster {
    log_template_tokens: Vec<String>,
    pub cluster_id: usize,
    pub size: usize,
    /// Original log messages stored as examples (up to max_examples)
    pub examples: Vec<String>,
}

impl Display for LogCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.log_template_tokens.join(" "))
    }
}

#[derive(Clone, Default)]
pub struct Node {
    key_to_child_node: HashMap<String, Node>,
    cluster_ids: Vec<usize>,
}

pub struct Drain {
    id_to_cluster: LruCache<usize, LogCluster>,

    max_node_depth: usize,

    /// Similarity threshold.
    /// A new log cluster will be created
    /// if the similarity of tokens for log message is below this.
    sim_th: f32,

    /// Maximum number of children within a node.
    max_children: usize,

    cluster_counter: usize,

    root: Node,

    param_str: String,

    revision: u64,

    /// Maximum number of example log messages to store per cluster
    max_examples: usize,
}

impl Debug for Drain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id_to_cluster: HashMap<_, _> = self
            .id_to_cluster
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        fn fmt_node(
            node: &Node,
            f: &mut std::fmt::Formatter<'_>,
            depth: usize,
            id_to_cluster: &HashMap<usize, LogCluster>,
        ) -> std::fmt::Result {
            for _ in 0..depth {
                write!(f, "  ")?;
            }
            writeln!(f, "Node {{ cluster_ids: {:?} }}", node.cluster_ids)?;
            for cluster_id in &node.cluster_ids {
                if let Some(cluster) = id_to_cluster.get(cluster_id) {
                    for _ in 0..depth + 1 {
                        write!(f, "  ")?;
                    }
                    writeln!(
                        f,
                        "id: {}, log_template_tokens: {:?}",
                        cluster.cluster_id, cluster.log_template_tokens
                    )?;
                }
            }
            for (key, child) in &node.key_to_child_node {
                for _ in 0..depth + 1 {
                    write!(f, "  ")?;
                }
                writeln!(f, "key: {}", key)?;
                fmt_node(child, f, depth + 1, id_to_cluster)?;
            }
            Ok(())
        }

        writeln!(f, "Drain {{")?;
        fmt_node(&self.root, f, 1, &id_to_cluster)?;
        writeln!(f, "}}")
    }
}

impl Default for Drain {
    fn default() -> Self {
        Self {
            id_to_cluster: LruCache::unbounded(),
            max_node_depth: 2,
            sim_th: 0.4,
            max_children: 100,
            cluster_counter: 0,
            root: Node::default(),
            param_str: "<*>".to_string(),
            revision: 0,
            max_examples: 0,
        }
    }
}

impl Drain {
    pub fn new(
        max_clusters: Option<usize>,
        max_node_depth: usize,
        sim_th: f32,
        max_children: usize,
        param_str: String,
        max_examples: usize,
    ) -> anyhow::Result<Self> {
        let id_to_cluster = match max_clusters {
            Some(max_clusters) => LruCache::new(NonZeroUsize::new(max_clusters).unwrap()),
            None => LruCache::unbounded(),
        };

        Ok(Self {
            id_to_cluster,
            max_node_depth,
            sim_th,
            max_children,
            cluster_counter: 0,
            root: Node::default(),
            param_str,
            revision: 0,
            max_examples,
        })
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn clusters(&self) -> Vec<&LogCluster> {
        self.id_to_cluster.iter().map(|(_, v)| v).collect()
    }

    pub fn train<T: AsRef<str>>(&mut self, log_message: T) -> LogCluster {
        let log_str = log_message.as_ref();
        let tokens = tokenize(log_str);
        let param_str = self.param_str.clone();
        let max_examples = self.max_examples;
        if let Some(cluster_id) = self.tree_search(&tokens, self.sim_th, false) {
            if let Some(cluster) = self.id_to_cluster.get_mut(&cluster_id) {
                cluster.log_template_tokens =
                    create_template(&param_str, &tokens, &cluster.log_template_tokens);
                cluster.size += 1;
                if cluster.examples.len() < max_examples {
                    cluster.examples.push(log_str.to_string());
                }
                self.revision += 1;
                return cluster.clone();
            }
        }

        self.cluster_counter += 1;
        let examples = if max_examples > 0 {
            vec![log_str.to_string()]
        } else {
            vec![]
        };
        let mut match_cluster = LogCluster {
            log_template_tokens: tokens,
            cluster_id: self.cluster_counter,
            size: 1,
            examples,
        };
        self.id_to_cluster
            .put(match_cluster.cluster_id, match_cluster.clone());
        self.add_seq_to_prefix_tree(&mut match_cluster);
        self.revision += 1;
        match_cluster
    }

    fn tree_search(
        &mut self,
        tokens: &[String],
        sim_th: f32,
        include_params: bool,
    ) -> Option<usize> {
        let token_count = tokens.len();

        let mut cur_node = self.root.key_to_child_node.get(&token_count.to_string())?;
        if token_count == 0 {
            return self
                .id_to_cluster
                .get(&cur_node.cluster_ids[0])
                .map(|c| c.cluster_id);
        }

        let mut cur_node_depth = 1;
        for token in tokens {
            // At max depth.
            if cur_node_depth == self.max_node_depth {
                break;
            }

            // At last token.
            if cur_node_depth == token_count {
                break;
            }

            cur_node = cur_node
                .key_to_child_node
                .get(token)
                .or_else(|| cur_node.key_to_child_node.get(&self.param_str))?;

            cur_node_depth += 1;
        }
        self.fast_match(
            &cur_node.cluster_ids.clone(),
            tokens,
            sim_th,
            include_params,
        )
    }
    fn fast_match(
        &mut self,
        cluster_ids: &[usize],
        tokens: &[String],
        sim_th: f32,
        include_params: bool,
    ) -> Option<usize> {
        let mut match_cluster = None;
        let param_str = self.param_str.clone();

        let mut max_sim = -1.0;
        let mut max_param_count = -1;
        for id in cluster_ids {
            let cluster = self.id_to_cluster.get(id);
            if let Some(cluster) = cluster {
                let (cur_sim, param_count) = get_seq_distance(
                    &param_str,
                    tokens,
                    &cluster.log_template_tokens,
                    include_params,
                );
                if cur_sim > max_sim || (cur_sim == max_sim && param_count > max_param_count) {
                    max_sim = cur_sim;
                    max_param_count = param_count;
                    match_cluster = Some(*id);
                }
            }
        }
        if max_sim >= sim_th {
            match_cluster
        } else {
            None
        }
    }

    fn add_seq_to_prefix_tree(&mut self, cluster: &mut LogCluster) {
        let token_count = cluster.log_template_tokens.len();
        let token_count_str = token_count.to_string();

        let mut cur_node: &mut Node = self
            .root
            .key_to_child_node
            .entry(token_count_str)
            .or_default();

        if token_count == 0 {
            cur_node.cluster_ids.push(cluster.cluster_id);
            return;
        }

        let mut current_depth = 1;
        for token in cluster.log_template_tokens.iter() {
            if current_depth >= self.max_node_depth || current_depth >= token_count {
                let mut new_cluster_ids = Vec::new();
                for cluster_id in cur_node
                    .cluster_ids
                    .iter()
                    .filter(|cluster_id| self.id_to_cluster.contains(cluster_id))
                {
                    new_cluster_ids.push(*cluster_id);
                }
                new_cluster_ids.push(cluster.cluster_id);
                cur_node.cluster_ids = new_cluster_ids;
                break;
            }

            if !cur_node.key_to_child_node.contains_key(token) {
                if !has_number(token) {
                    if cur_node.key_to_child_node.contains_key(&self.param_str) {
                        if cur_node.key_to_child_node.len() < self.max_children {
                            let new_node = Node::default();
                            cur_node.key_to_child_node.insert(token.clone(), new_node);
                            cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
                        } else {
                            cur_node = cur_node.key_to_child_node.get_mut(&self.param_str).unwrap();
                        }
                    } else if cur_node.key_to_child_node.len() + 1 < self.max_children {
                        let new_node = Node::default();
                        cur_node.key_to_child_node.insert(token.clone(), new_node);
                        cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
                    } else if cur_node.key_to_child_node.len() + 1 == self.max_children {
                        let new_node = Node::default();
                        cur_node
                            .key_to_child_node
                            .insert(self.param_str.clone(), new_node);
                        cur_node = cur_node.key_to_child_node.get_mut(&self.param_str).unwrap();
                    } else {
                        cur_node = cur_node.key_to_child_node.get_mut(&self.param_str).unwrap();
                    }
                } else if !cur_node.key_to_child_node.contains_key(&self.param_str) {
                    let new_node = Node::default();
                    cur_node
                        .key_to_child_node
                        .insert(self.param_str.clone(), new_node);
                    cur_node = cur_node.key_to_child_node.get_mut(&self.param_str).unwrap();
                } else {
                    cur_node = cur_node.key_to_child_node.get_mut(&self.param_str).unwrap();
                }
            } else {
                cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
            }

            current_depth += 1;
        }
    }
}
fn get_seq_distance(
    param_str: &str,
    seq1: &[String],
    seq2: &[String],
    include_params: bool,
) -> (f32, isize) {
    let mut sim_tokens = 0;
    let mut param_count = 0;

    for (token1, token2) in seq1.iter().zip(seq2.iter()) {
        if token1 == param_str {
            param_count += 1;
        } else if token1 == token2 {
            sim_tokens += 1;
        }
    }
    if include_params {
        sim_tokens += param_count;
    }
    (sim_tokens as f32 / seq1.len() as f32, param_count)
}

fn create_template(param_str: &str, seq1: &[String], seq2: &[String]) -> Vec<String> {
    seq1.iter()
        .zip(seq2.iter())
        .map(|(token1, token2)| {
            if token1 == token2 {
                token1.clone()
            } else {
                param_str.to_string()
            }
        })
        .collect()
}
fn has_number(s: &str) -> bool {
    s.chars().any(|c| c.is_numeric())
}

fn tokenize(log_message: &str) -> Vec<String> {
    log_message
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    mod train {
        use super::*;

        #[test]
        fn test() {
            let logs = vec![
                "connected to 10.0.0.1",
                "connected to 10.0.0.2",
                "connected to 10.0.0.3",
                "Hex number 0xDEADBEAF",
                "Hex number 0x10000",
                "user davidoh logged in",
                "user eranr logged in",
            ];
            let mut drain = Drain::default();
            for log in logs {
                drain.train(log);
            }
            let mut clusters = drain.clusters();
            clusters.sort_by_key(|c| c.cluster_id);
            assert_eq!(
                clusters,
                vec![
                    &LogCluster {
                        log_template_tokens: vec![
                            String::from("connected"),
                            String::from("to"),
                            String::from("<*>"),
                        ],
                        cluster_id: 1,
                        size: 3,
                        examples: vec![],
                    },
                    &LogCluster {
                        log_template_tokens: vec![
                            String::from("Hex"),
                            String::from("number"),
                            String::from("<*>"),
                        ],
                        cluster_id: 2,
                        size: 2,
                        examples: vec![],
                    },
                    &LogCluster {
                        log_template_tokens: vec![
                            String::from("user"),
                            String::from("<*>"),
                            String::from("logged"),
                            String::from("in"),
                        ],
                        cluster_id: 3,
                        size: 2,
                        examples: vec![],
                    },
                ]
            );
        }
    }

    #[test]
    fn revision_increments_on_train_and_reuse() {
        let mut drain = Drain::default();
        assert_eq!(drain.revision(), 0);
        let first = drain.train("connected to 1.1.1.1");
        assert_eq!(drain.revision(), 1);
        assert_eq!(first.size, 1);

        // Reuse same cluster, should increment revision and size
        let second = drain.train("connected to 2.2.2.2");
        assert_eq!(drain.revision(), 2);
        assert_eq!(second.cluster_id, first.cluster_id);
        assert_eq!(second.size, 2);
    }

    #[test]
    fn template_uses_wildcards_for_varying_tokens() {
        let mut drain = Drain::default();
        drain.train("user alice logged in");
        let updated = drain.train("user bob logged in");
        assert_eq!(
            updated.log_template_tokens,
            vec!["user", "<*>", "logged", "in"]
        );
    }

    #[test]
    fn get_seq_distance_counts_params_only_when_included() {
        let seq1 = vec!["a".into(), "<*>".into(), "c".into()];
        let seq2 = vec!["a".into(), "b".into(), "c".into()];

        let (sim_without_params, params) = get_seq_distance("<*>", &seq1, &seq2, false);
        assert_eq!(params, 1);
        // Matches a and c only -> 2/3
        assert!((sim_without_params - (2.0 / 3.0)).abs() < f32::EPSILON);

        let (sim_with_params, params2) = get_seq_distance("<*>", &seq1, &seq2, true);
        assert_eq!(params2, 1);
        // With params, wildcard counts as match so 3/3
        assert!((sim_with_params - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn examples_are_stored_up_to_max() {
        let mut drain = Drain::new(None, 2, 0.4, 100, "<*>".to_string(), 2).unwrap();

        drain.train("connected to 10.0.0.1");
        drain.train("connected to 10.0.0.2");
        drain.train("connected to 10.0.0.3"); // Should not be stored (max_examples=2)

        let clusters = drain.clusters();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].examples.len(), 2);
        assert_eq!(clusters[0].examples[0], "connected to 10.0.0.1");
        assert_eq!(clusters[0].examples[1], "connected to 10.0.0.2");
    }

    #[test]
    fn examples_empty_when_max_is_zero() {
        let mut drain = Drain::new(None, 2, 0.4, 100, "<*>".to_string(), 0).unwrap();

        drain.train("connected to 10.0.0.1");
        drain.train("connected to 10.0.0.2");

        let clusters = drain.clusters();
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].examples.is_empty());
    }

    #[test]
    fn max_clusters_evicts_oldest() {
        // Only allow 2 clusters, use high similarity threshold to prevent merging
        let mut drain = Drain::new(Some(2), 2, 0.9, 100, "<*>".to_string(), 0).unwrap();

        // Use very different messages to ensure they create separate clusters
        drain.train("alpha one two three");
        drain.train("beta four five six");
        drain.train("gamma seven eight nine"); // Should evict "alpha" cluster

        let clusters = drain.clusters();
        assert_eq!(clusters.len(), 2);

        // Check that we have beta and gamma, not alpha
        let templates: Vec<String> = clusters.iter().map(|c| c.to_string()).collect();
        assert!(!templates.iter().any(|t| t.contains("alpha")));
        assert!(templates.iter().any(|t| t.contains("beta")));
        assert!(templates.iter().any(|t| t.contains("gamma")));
    }

    #[test]
    fn empty_log_creates_cluster() {
        let mut drain = Drain::default();
        let cluster = drain.train("");

        assert_eq!(cluster.size, 1);
        assert!(cluster.log_template_tokens.is_empty());
    }

    #[test]
    fn whitespace_only_log_creates_empty_cluster() {
        let mut drain = Drain::default();
        let cluster = drain.train("   \t  ");

        assert_eq!(cluster.size, 1);
        assert!(cluster.log_template_tokens.is_empty());
    }

    #[test]
    fn similar_logs_with_numbers_cluster_together() {
        let mut drain = Drain::default();

        drain.train("request took 100ms");
        drain.train("request took 200ms");
        let cluster = drain.train("request took 50ms");

        // All should be in same cluster with wildcard for the number
        assert_eq!(cluster.size, 3);
        assert_eq!(cluster.to_string(), "request took <*>");
    }

    #[test]
    fn custom_param_str_is_used() {
        let mut drain = Drain::new(None, 2, 0.4, 100, "???".to_string(), 0).unwrap();

        drain.train("user alice logged in");
        let cluster = drain.train("user bob logged in");

        assert_eq!(cluster.to_string(), "user ??? logged in");
    }
}
