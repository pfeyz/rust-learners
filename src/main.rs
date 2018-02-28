pub mod colag;

extern crate rand;

use std::mem;
use std::fmt;
use std::collections::{HashSet};
use std::thread;
use std::sync::Arc;

use rand::Rng;
use rand::distributions::{Range, Sample};

use colag::{Domain, NUM_PARAMS};

const COLAG_TSV: &'static str = "./COLAG_2011_ids.txt";
const NUM_SENTENCES: u32 = 2_000_000;
const RUNS_PER_LANGUAGE: u8 = 100;

const LEARNING_RATE: f64 = 0.001;
const THRESHOLD: f64 = 0.02;

type Grammar = u16;
type Sentence = u32;
type ParamWeights = [f64; NUM_PARAMS];

enum Hypothesis {
    Trigger ( Grammar ),
    Genetic ( HashSet<Grammar> ),
    RewardOnlyVL ( ParamWeights ),
    RewardOnlyRelevantVL ( ParamWeights ),
}

impl fmt::Display for Hypothesis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Hypothesis::RewardOnlyRelevantVL(weights) | &Hypothesis::RewardOnlyVL(weights) => {
                write!(f, "VL( ")?;
                for w in weights.iter(){
                    write!(f, "{:.2} ", w)?;
                }
                write!(f, ")")
            }
            _ => write!(f, "---")
        }
    }
}

fn init_weights() -> ParamWeights {
    unsafe {
        let mut array: ParamWeights = mem::uninitialized();
        for param in 0..NUM_PARAMS {
            array[param] = 0.5;
        }
        array
    }
}

impl Hypothesis {
    fn new_trigger() -> Hypothesis {
        Hypothesis::Trigger(0)
    }

    fn new_reward_only() -> Hypothesis {
        Hypothesis::RewardOnlyVL(init_weights())
    }

    fn new_reward_only_relevant() -> Hypothesis {
        Hypothesis::RewardOnlyRelevantVL(init_weights())
    }

    fn new_genetic() -> Hypothesis {
        Hypothesis::Genetic(HashSet::new())
    }

    fn converged(&self) -> bool {
        match *self {
            Hypothesis::RewardOnlyVL(weights) | Hypothesis::RewardOnlyRelevantVL(weights) => {
                for weight in weights.iter() {
                    if (weight > &THRESHOLD) & (weight < &(1.0 - THRESHOLD)) {
                        return false;
                    }
                }
                true
            }
            _ => false
        }
    }

}

struct IllegalGrammar ( Grammar );

fn random_weighted_grammar(weights: &ParamWeights) -> Grammar {
    let mut grammar = 0;
    for param in 0..NUM_PARAMS {
        if weighted_coin_flip(weights[param]) {
            grammar = set_param(grammar, param);
        }
    }
    grammar
}

fn sentence_parses(domain: &Domain, grammar: &Grammar, sentence: &Sentence) -> Result<bool, IllegalGrammar> {
    if let Some(sentences) = domain.language.get(grammar) {
        Ok(sentences.contains(sentence))
    } else {
        Err(IllegalGrammar(*grammar))
    }
}

pub fn reward_weights(mut weights: ParamWeights, grammar: &Grammar, _: &Sentence) -> ParamWeights {
    for param in 0..NUM_PARAMS {
        let weight = weights[param];
        if get_param(grammar, param) == 0 {
            weights[param] -= LEARNING_RATE * weight
        } else {
            weights[param] += LEARNING_RATE * (1. - weight)
        }
    }
    weights
}

pub fn reward_relevant_weights(mut weights: ParamWeights, grammar: &Grammar, sentence: &Sentence, _triggers: ()) -> ParamWeights {
    for param in 0..NUM_PARAMS {
        let weight = weights[param];
        if get_param(grammar, param) == 0 {
            weights[param] -= LEARNING_RATE * weight
        } else {
            weights[param] += LEARNING_RATE * (1. - weight)
        }
    }
    weights
}

fn consume_sentence(hypothesis: Hypothesis, domain: &Domain, sentence: &Sentence) -> Hypothesis {
    match hypothesis {
        Hypothesis::RewardOnlyVL(mut weights) => {
            loop {
                let ref grammar = random_weighted_grammar(&weights);
                // only returns ok if grammar exists in colag
                if let Ok(parses) = sentence_parses(domain, grammar, sentence) {
                    if parses {
                        weights = reward_weights(weights, grammar, sentence);
                    }
                    break;
                }
            }
            Hypothesis::RewardOnlyVL(weights)
        },
        Hypothesis::RewardOnlyRelevantVL(weights) => {
            loop {
                let ref grammar = random_weighted_grammar(&weights);
                // only returns ok if grammar exists in colag
                if let Ok(parses) = sentence_parses(domain, grammar, sentence) {
                    if parses {
                        reward_relevant_weights(weights, grammar, sentence, ());
                    }
                    break;
                }
            }
            Hypothesis::RewardOnlyRelevantVL(weights)
        },
        _ => panic!("not implemented")
    }
}

/// Returns paramter # `param_num` from `grammar`.
fn get_param(grammar: &Grammar, param_num: usize) -> Grammar {
    (grammar >> (NUM_PARAMS - param_num - 1)) & 1
}

/// Returns `grammar` with `param_num` turned on.
fn set_param(grammar: Grammar, param_num: usize) -> Grammar {
    grammar + (1 << (NUM_PARAMS - param_num - 1))
}

/// Returns true `weight` percent of the time
fn weighted_coin_flip(weight: f64) -> bool {
    debug_assert!((weight >= 0.) & (weight <= 1.));
    let mut rng = rand::thread_rng();
    let mut range = Range::new(0., 1.);
    range.sample(&mut rng) < weight
}

struct Report {
    hypothesis: Hypothesis,
    target: Grammar,
    converged: bool,
    consumed: u32
}

fn learn_language(colag: &Domain, target: &Grammar, mut hypothesis: Hypothesis) -> Report {
    let mut rng = rand::thread_rng();
    let sentences: Vec<&u32> = colag.language
        .get(&target)
        .expect(&format!("language {} does not exist", target))
        .iter().collect();
    let mut converged = false;
    let mut consumed = NUM_SENTENCES;
    for i in 1..NUM_SENTENCES {
        let sentence = rng.choose(&sentences).unwrap();
        hypothesis = consume_sentence(hypothesis, colag, &sentence);
        converged = hypothesis.converged();
        if converged {
            consumed = i;
            break;
        }
    }
    Report {
        hypothesis: hypothesis,
        target: *target,
        consumed: consumed,
        converged: converged
    }
}

fn main() {
    let colag = Arc::new(Domain::from_file(&String::from(COLAG_TSV)).unwrap());
    let languages = vec![611, 584, 2253, 3856];
    let mut handles = Vec::new();

    for target in languages {
        let colag = colag.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let mut hypothesis = Hypothesis::new_reward_only();
                let report = learn_language(&colag, &target, hypothesis);
                println!("{} {} {} {}", report.converged, report.consumed, report.target, report.hypothesis)
            }
        }))
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

enum Parameter {
    SP,
    HIP,
    HCP,
    OPT,
    NS,
    NT,
    WHM,
    PI,
    TM,
    VtoI,
    ItoC,
    AH,
    QInv
}
