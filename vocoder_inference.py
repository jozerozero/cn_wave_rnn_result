import argparse
from pathlib import Path
import numpy as np
from vocoder import inference as vocoder
import librosa
from glob import glob
from tqdm import tqdm

def synthesize_and_save(inpath, outpath):
    spec=np.load(inpath).T
    generated_wav = vocoder.infer_waveform(spec)
    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, 16000), mode="constant")
    # Save it on the disk
    librosa.output.write_wav(outpath, generated_wav.astype(np.float32), 16000)
    pass

#python vocoder_inference.py -i ../vox1_test/wav/ -o spkid --gpu_str 0
if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        #default="vocoder/saved_models/pretrained/pretrained.pt",
                        default="gta_model/gta_model/gta_model.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-i", "--input", type=str, required=True, help="input data(pickle) dir")
    parser.add_argument("-o", "--output", type=str, default='spkids', help="output data dir")
    parser.add_argument('--gpu_str', default='0')
    args = parser.parse_args()
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_str)

    is_dir=os.path.isdir(args.input)

    print("Preparing the vocoder...")
    vocoder.load_model(args.voc_model_fpath)

    speaker_max=10
    speaker_cnt=0
    prev_speaker=''
    if is_dir:
        mel_files=glob("%s/**/mel-*.npy" % args.input, recursive=True)
        #import pdb;pdb.set_trace()
        for mel_file in tqdm(mel_files):
            fn=os.path.basename(mel_file)
            cur_speaker=fn.split('-')[1]
            cur_speaker=fn.split('_')[0]
            if cur_speaker==prev_speaker:
                speaker_cnt+=1
                if speaker_cnt>speaker_max:
                    continue
            if cur_speaker!=prev_speaker:
                speaker_cnt=1
            prev_speaker=cur_speaker
            dn=os.path.dirname(mel_file)
            fn=fn[4:]
            fn=fn[:-3]+'wav'
            out_file=os.path.join(dn, fn)
            synthesize_and_save(mel_file, out_file)
            print(out_file)

    else:
        ## Generating the waveform
        print("Synthesizing the waveform:")
        synthesize_and_save(args.input, args.output)
