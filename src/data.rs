use ndarray::Array1;
use std::fs::File;
use std::io::Read;

fn load_and_decompress(filepath: &String) -> String {
    let file = File::open(filepath).unwrap();
    //let mut buffer = Vec::new();
    //BufReader::new(file).read_to_end(&mut buffer).unwrap();
    let mut zip_archive = zip::ZipArchive::new(file).unwrap();
    let mut file = zip_archive.by_index(0).unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data);
    data
}

fn vectorized_result(num: usize) -> Array1<f64> {
    let vect = Array1::zeros(10);
    vect[num] = 1.0;
    vect
}
