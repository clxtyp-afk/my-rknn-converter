from rknn.api import RKNN
import os

def convert():
    rknn = RKNN(verbose=True)
    # 配置针对 RK3588
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    
    print('--> Loading ONNX model')
    if rknn.load_onnx(model='yolo11n.onnx') != 0:
        print('Load failed'); return

    print('--> Building RKNN model (No Quantization for high success rate)')
    # 先关闭量化以保证 100% 转换成功，以后熟悉了再加 dataset.txt 开启量化
    if rknn.build(do_quantization=False) != 0:
        print('Build failed'); return

    print('--> Exporting RKNN model')
    if rknn.export_rknn('yolo11n.rknn') != 0:
        print('Export failed'); return
    
    rknn.release()

if __name__ == '__main__':
    convert()
