import os
import shutil
def mvfile(source_dir, destination_dir, i):
    # times = str(update_times*300)
    files = os.listdir(source_dir)
    for file in files:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        if i in source_file:
            shutil.move(source_file, destination_file)


def mvfile_filename(source_dir, destination_dir, filename):
    shutil.move(source_dir + '/' + filename, destination_dir + '/' + filename)
