from os import path, mkdir
from glob import glob
import subprocess
from utils.common_utils import  check_file

def video2audio(pv_address):
    ll_video_files = glob(path.join(pv_address, '*' , '*.mp4'))
    lv_base_save_address = path.join(path.dirname(pv_address),'audio_data')

    for i,f in enumerate(ll_video_files):
        lv_dir_name = path.basename(path.dirname(f))
        lv_file_name = path.splitext(path.basename(f))[0] + '.wav'
        lv_save_address = path.join(lv_base_save_address, lv_dir_name, lv_file_name)
        check_file(path.dirname(lv_save_address))
        command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(f, lv_save_address)
        subprocess.call(command, shell=True)

if __name__ == '__main__':
    lv_load_address = '/media/shini/D/hmburg_competation/data/video_data'
    video2audio(lv_load_address)