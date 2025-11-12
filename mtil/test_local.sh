python -m src.main \
    --load ckpt/DTD/MNIST/MNIST.pth \
    --save ckpt/DTD/MNIST \
    --eval-single data/single_images/play-6865967_1280.jpg \
    --class-names data/text_classes/imagenet_classes.txt \
    --eval-only