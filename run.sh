# please download the data before run

rm -rf data
mkdir data
matlab -nodisplay -nosplash -r "read_physionet(); quit"
python3 mina.py