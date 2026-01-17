python -m src.main \
    --load ckpt/DTD/MNIST/MNIST.pth \
    --save data/single_images/results \
    --eval-single data/single_images/Artomyces_pyxidatus.jpg \
    --class-names data/text_classes/imagenet_classes.txt \
    --eval-only