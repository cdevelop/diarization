cd ../bin
make
cd ../diarization
# ./diarization

for j in $(ls result_rttm); do
	cat result_rttm/$j;
done > result.rttm

for j in $(ls result_mdtm); do
	cat result_mdtm/$j;
done > result.mdtm

./md-eval.pl -1 -c 0.25 -r ref.rttm -s 2>log.txt result.rttm > DER.txt
