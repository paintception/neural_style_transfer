source activate PhD
KERAS_BACKEND=tensorflow 

START=64
END=512
BATCH=64
CONTENT_IMG="../images/mat.jpg"
STYLE_IMG="../images/munch.jpg"

for ((i<=START;i<=END;i+=BATCH));
    do
        start=$((i-BATCH))
        end=$i
        nohup python SolutionAnalysis.py --start_batch $start --end_batch $end --content_img $CONTENT_IMG --style_img $STYLE_IMG &
	wait
    done
