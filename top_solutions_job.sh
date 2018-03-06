source activate PhD

KERAS_BACKEND=tensorflow

content_img=../images/mat.jpg
style_img=../images/munch.jpg
normalizer=False

nohup python top_solutions_optimizer.py --content_img $content_img --style_img $style_img --normalizer $normalizer 
