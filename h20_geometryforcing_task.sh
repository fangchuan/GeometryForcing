arena --loglevel info submit pytorch  \
    --toleration 'all' \
    --namespace mri \
    --name zhenqing-debug-geoforce2 \
    --gpus 1 \
    --workers 1 \
    --cpu 24 \
    --memory 100Gi \
    --hostNetwork true \
    --data-dir /data-nas:/data-nas \
    --data-dir /data-high-nas:/data-high-nas \
    --image registry.qunhequnhe.com/mri/geometryforcing:v0 \
    --image-pull-policy  IfNotPresent \
    --working-dir=/workspace \
    --sync-mode git \
    --sync-source https://gitlab.qunhequnhe.com/zhenqing/Spatial3DV.git \
    --sync-branch develop \
    --clean-task-policy None \
    --shell "bash" \
    "env;
cd /data-nas/data/experiments/zhenqing/GeometryForcing;
echo \$PWD;

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple;

export INCEPTION_PRETRAINED_MODEL_PATH=/data-nas/data/experiments/zhenqing/cache/weights-inception-2015-12-05-6726825d.pth;

mkdir -p /root/.cache/torch/hub/checkpoints;
cp \$INCEPTION_PRETRAINED_MODEL_PATH /root/.cache/torch/hub/checkpoints/;

sleep 344h
"
