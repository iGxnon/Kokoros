use lazy_static::lazy_static;
use std::collections::HashMap;

pub fn get_vocab() -> std::collections::HashMap<char, usize> {
    let pad = "$";
    let punctuation = ";:,.!?¡¿—…\"«»“” ";
    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    let symbols: String = [pad, punctuation, letters, letters_ipa].concat();

    symbols
        .chars()
        .enumerate()
        .collect::<HashMap<_, _>>()
        .into_iter()
        .map(|(idx, c)| (c, idx))
        .collect()
}

pub fn get_reverse_vocab() -> HashMap<usize, char> {
    VOCAB.iter().map(|(&c, &idx)| (idx, c)).collect()
}

lazy_static! {
    pub static ref VOCAB: HashMap<char, usize> = get_vocab();
    pub static ref REVERSE_VOCAB: HashMap<usize, char> = get_reverse_vocab();
}
