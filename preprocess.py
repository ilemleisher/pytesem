import numpy as np
import os, h5py, argparse
from dev.utils import get_files
from preprocessing.utils import downsample, chunk, fft, filter_chunks, preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw data files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing the .hdf5 files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder where preprocessed files will be saved.')
    parser.add_argument('--downsample_factor', type=int, default=10, help='Downsample reduction factor.')
    parser.add_argument('--n_chunks', type=int, default=1, help='Number of chunks to divide each data file into.')
    parser.add_argument('--sampling_rate', type=float, default=1.25e6, help='Sampling rate of the raw data in Hz.')
    parser.add_argument('--channel_number', type=int, default=0, help='Channel number to read from the raw data file (0-indexed).')
    return parser.parse_args()

def main():

    args = parse_args()
    
    n_chunks = args.n_chunks
    downsample_factor = args.downsample_factor
    sampling_rate = args.sampling_rate
    channel_number = args.channel_number
    output_path = args.output_dir
    input_path = args.input_dir

    os.makedirs(output_path, exist_ok=True)

    filenames = get_files(input_path)
    
    # Loop over each file in the folder
    for filename in filenames:
        filepath = os.path.join(input_path, filename)
        print(f"Reading: {filename}")

        with h5py.File(filepath, "r") as data:
            events = data['adc1'].keys()
            print("Found", len(events), "events")

            # Data containers
            freqs_list, asd_list = [], []

            # Loop over each event in the file
            for event in events:

                # Read ADC1 output from the specified channel
                waveforms = np.array(data["adc1"][str(event)])
                n_samples = waveforms.shape[1]
                time_data = np.arange(n_samples) / sampling_rate
                waveform_data = waveforms[channel_number]

                # Calculations
                target_len = n_samples // downsample_factor
                duration = n_samples // sampling_rate
                fft_fs = target_len // duration

                # Downsample the raw waveform data
                new_tdata, new_data = preprocess(time_data, waveform_data, target_len)

                # Divide the preprocessed data into chunks
                chunks = chunk(new_tdata, new_data, n_chunks)

                # Loop over each chunk
                for data_chunk in chunks:

                    # Compute the ASD for each chunk
                    freqs, asd = fft(data_chunk[1], fft_fs)

                    freqs_list.append(freqs.astype(np.float32))
                    asd_list.append(asd.astype(np.float32))

        print(f'Saving...')

        # Stack data and save to .npz file
        np.savez(
            f"{output_path}/{filename[:-5]}.npz",
            freqs_list=np.stack(freqs_list),
            asd_list=np.stack(asd_list),
)
        print(f'Saved.')

if __name__ == '__main__':
    main()