import os
from tqdm import tqdm
import numpy as np

print("Start spliting wav...")

wav_scp = open('wav.scp')
for wav_pos in tqdm(wav_scp.readlines()):
    name, pos = wav_pos.split()
    wavfile = open(pos, 'rb')
    binary_data = np.array(list(wavfile.read()))
    wavfile.close()

    cover_dict = {}
    rttm = open('result_rttm/{}.rttm'.format(name), 'r')
    os.system('mkdir -p result_wav/' + name)

    for line in rttm.readlines():
        temp  = line.split()
        start = int(float(temp[3]) * 8000)*2
        end   = int((float(temp[3]) + float(temp[4])) * 8000)*2
        ident = int(temp[7])
        if (ident not in cover_dict.keys()):
            cover_dict[ident] = np.zeros(len(binary_data), dtype = int)
            cover_dict[ident][0:44] = 1
        cover_dict[ident][start+44:end+44] = 1

    for key in cover_dict.keys():
        output = bytes(list(cover_dict[key] * binary_data))
        outfile = open('result_wav/{}/{}.wav'.format(name, key), 'wb')
        outfile.write(output)
        outfile.close()

