use ndarray::Array1;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use std::fs::File;
use std::io;
use std::io::Read;
pub type VectorLabelPixelsPair = (Array1<f64>, Array1<f64>);
pub type LabelPixelsPair = (u8, Array1<f64>);

fn get_decompressed_data(filepath: &String) -> Result<Vec<u8>, io::Error> {
    let file = File::open(filepath)?;
    let mut zip_archive = zip::ZipArchive::new(file)?;
    let mut file = zip_archive.by_index(0)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    Ok(data)
}

fn split_and_format_dataset(
    dataset: Vec<LabelPixelsPair>,
) -> (Vec<VectorLabelPixelsPair>, Vec<LabelPixelsPair>) {
    let (train, test) = dataset.split_at(dataset.len() * 9 / 10);
    let train = train
        .iter()
        .map(|(label, values)| -> (Array1<f64>, Array1<f64>) {
            let label_vector = vectorize_label(label.to_usize().expect("Couldn't convert value!"));
            (label_vector, values.to_owned())
        })
        .collect();
    (train, test.to_vec())
}

pub fn get_data(
    filepath: &String,
) -> Result<(Vec<VectorLabelPixelsPair>, Vec<LabelPixelsPair>), Box<dyn std::error::Error>> {
    let decompressed_data = get_decompressed_data(filepath)?;
    let mut reader = csv::Reader::from_reader(decompressed_data.as_slice());
    let data: Result<Vec<(u8, Array1<f64>)>, _> = reader
        .records()
        .map(
            |record_result| -> Result<LabelPixelsPair, Box<dyn std::error::Error>> {
                let record = record_result?;
                let mut record_iterator = record.iter();
                let label = record_iterator.next().unwrap().parse::<u8>()?;
                let pixel_values: Result<Array1<f64>, Box<dyn std::error::Error>> = record_iterator
                    .map(|record_value| -> Result<f64, _> {
                        let brightness_int = record_value.parse::<u8>()?;
                        Ok(convert_brightness_to_fraction(brightness_int))
                    })
                    .collect();
                Ok((label, pixel_values?))
            },
        )
        .collect();
    Ok(split_and_format_dataset(data?))
}
fn vectorize_label(num: usize) -> Array1<f64> {
    let mut vect = Array1::zeros(10);
    vect[num] = 1.0;
    vect
}
fn convert_brightness_to_fraction(brightness: u8) -> f64 {
    brightness.to_f64().expect("Error converting int to float!") / 255.0
}
