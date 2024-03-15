
MODEL='yolox_tiny'

mkdir -p ~/.cache/thesis/yolox
wget "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$MODEL.pth" -nc -O "$HOME/.cache/thesis/yolox/$MODEL.pth"

