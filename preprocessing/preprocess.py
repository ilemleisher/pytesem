import h5py, sys, os
import numpy as np
from preprocessing_utils import downsample, chunk, fft, filter_chunks
sys.path.append("/home/ilemleisher/em_project/dev/") 
from utils import get_files

if __name__ == '__main__':

    # Variables
    n_chunks = 2                        # Number of chunks to divide each data file into
    post_downsample_length = 12000      # Target length of the raw data file for downsampling
    sampling_rate = 1.25e6              # Sampling rate of the raw data in Hz
    channel_number = 0                  # Channel number to read from the raw data file (0-indexed)

    # Path to the folder containing the .hdf5 files
    base_path = "/data/lbl/run21/raw/"
    folder = "continuous_I4_D20250102_T224744/"

    output_path = '/home/ilemleisher/data/'
    output_directory = os.path.join(output_path, folder)
    os.makedirs(output_directory, exist_ok=True)

    path = os.path.join(base_path, folder)
    filenames = get_files(path)
    
    # Loop over each file in the folder
    for filename in filenames:
        filepath = path+filename
        print(f"Reading: {filename}")
        num_filtered = 0
        num_chunks = 0

        with h5py.File(filepath, "r") as data:
            events = data['adc1'].keys()
            print("Found", len(events), "events")

            # Data containers
            freqs_list, asd_list, waveform_data_list = [], [], []
            new_tdata_list, new_data_list, time_data_list = [], [], []

            # Loop over each event in the file
            for event in events:

                # Read ADC1 output from the specified channel
                waveforms = np.array(data["adc1"][str(event)])
                time_data = np.arange(waveforms.shape[1]) / sampling_rate
                waveform_data = waveforms[channel_number]

                # Downsample the raw waveform data
                new_tdata, new_data = downsample(time_data, waveform_data, post_downsample_length)

                # Divide the raw data into chunks
                chunks = chunk(new_tdata, new_data, n_chunks)

                # Count number of chunks
                num_chunks += len(chunks)

                # Discard any chunks that contains peaks above 4 sigma
                filtered_chunks, num_filtered_chunks = filter_chunks(chunks, 4)

                # Count number of filtered chunks
                num_filtered += num_filtered_chunks

                # Loop over each remaining chunk
                for data_chunk in filtered_chunks:

                    # Compute the ASD for each chunk
                    freqs, asd = fft(data_chunk[1])

                    freqs_list.append(freqs.astype(np.float32))
                    asd_list.append(asd.astype(np.float32))
                    waveform_data_list.append(waveform_data.astype(np.float32))
                    new_tdata_list.append(new_tdata.astype(np.float32))
                    new_data_list.append(new_data.astype(np.float32))
                    time_data_list.append(time_data.astype(np.float32))

        print(f"Filtered out {num_filtered}/{num_chunks} chunks in file {filename} ({num_filtered/num_chunks*100:.1f}% removal rate)")

        print(f'Saving...')

        # Stack data and save to .npz file
        np.savez_compressed(
            f"{output_directory}/{filename[:-5]}.npz",
            freqs_list=np.stack(freqs_list),         # (N, F) or (F,)
            asd_list=np.stack(asd_list),             # (N, F)
            # optional metadata
            new_tdata_list=np.stack(new_tdata_list),
            new_data_list=np.stack(new_data_list),
            time_data_list=np.stack(time_data_list),
            waveform_data_list=np.stack(waveform_data_list)
)
        print(f'Saved.')
