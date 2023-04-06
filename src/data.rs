use ndarray::Array1;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use std::fs::File;
use std::io;
use std::io::Read;

fn get_decompressed_data(filepath: &String) -> Result<Vec<u8>, io::Error> {
    let file = File::open(filepath)?;
    let mut zip_archive = zip::ZipArchive::new(file)?;
    let mut file = zip_archive.by_index(0)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    Ok(data)
}
pub fn get_data(
    filepath: &String,
) -> Result<Vec<(Array1<f64>, Array1<f64>)>, Box<dyn std::error::Error>> {
    let decompressed_data = get_decompressed_data(filepath)?;
    let mut reader = csv::Reader::from_reader(decompressed_data.as_slice());
    let data: Result<Vec<(Array1<f64>, Array1<f64>)>, _> = reader
        .records()
        .map(
            |record_result| -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>> {
                let record = record_result?;
                let mut record_iterator = record.iter();
                let label_vector =
                    vectorized_result(record_iterator.next().unwrap().parse::<usize>()?);
                let pixel_values: Result<Array1<f64>, Box<dyn std::error::Error>> = record_iterator
                    .map(|record_value| -> Result<f64, _> {
                        let brightness_int = record_value.parse::<u8>()?;
                        Ok(convert_brightness_to_fraction(brightness_int))
                    })
                    .collect();
                Ok((label_vector, pixel_values?))
            },
        )
        .collect();
    Ok(data?)
}
fn vectorized_result(num: usize) -> Array1<f64> {
    let mut vect = Array1::zeros(10);
    vect[num] = 1.0;
    vect
}
fn convert_brightness_to_fraction(brigthness: u8) -> f64 {
    brigthness.to_f64().unwrap() / 255.0
}
