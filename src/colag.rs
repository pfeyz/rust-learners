extern crate csv;
use std::error::Error;
use std::collections::{HashMap,HashSet};

pub type Grammar = u16;
pub const NUM_PARAMS: usize = 13;

#[derive(Debug)]
pub struct Domain {
    pub language: HashMap<Grammar, HashSet<u32>>
}

type Record = (u16, u32, u32);

impl Domain {
    pub fn new() -> Domain {
        let lang = HashMap::new();
        Domain { language: lang }
    }
    pub fn from_file(filename: &String) -> Result<Domain, Box<Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)
            .expect(filename);
        let mut domain = Domain::new();

        for result in rdr.deserialize() {
            let (grammar, sentence, _tree): Record = result?;
            if domain.language.contains_key(&grammar){
                domain.language.get_mut(&grammar).map(|set| set.insert(sentence));
            } else {
                let mut set = HashSet::new();
                set.insert(sentence);
                domain.language.insert(grammar, set);
            }
        }
        assert!(domain.language.len() == 3072, "Expected 3072 languages in Colag");
        {
            let english = domain.language.get(&611).unwrap();
            assert!(english.len() == 360, "Expected 360 sentences in Colag English");
            for s in vec![3138, 1970, 5871, 6923, 1969].iter() {
                assert!(english.contains(&s), format!("Expected sentence {} in Colag English", &s))
            }
        }
        Ok(domain)
    }
}
