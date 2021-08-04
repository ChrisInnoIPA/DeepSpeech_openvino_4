"""
author: Wira D K Putra
25 February 2020

See original repo at
https://github.com/WiraDKP/pytorch_speaker_embedding_for_diarization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import onnx
from model.src.model import Encoder
from model.src.dataset import BaseLoad
from model.src.utils import zcr_vad, get_timestamp
from model.src.cluster import OptimizedAgglomerativeClustering
from openvino.inference_engine import IECore, IENetwork
from pydub import AudioSegment
from pydub.silence import split_on_silence
from speekercut import get_wav_make
import shutil

class BasePredictor(BaseLoad):
    def __init__(self, config_path, max_frame, hop):
        config = torch.load(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(config.sr, config.n_mfcc)
        self.ndim = config.ndim
        self.max_frame = max_frame
        self.hop = hop
        
    @staticmethod
    def _plot_diarization(y, spans, speakers):
        c = y[0].cpu().numpy().copy()
        for (start, end), speaker in zip(spans, speakers):
            c[start:end] = speaker

        plt.figure(figsize=(15, 2))
        ax = plt.axes()
        ax.plot(y[0], "k-")
        for idx, speaker in enumerate(set(speakers)):
            ax.fill_between(range(len(c)), -1, 1, where=(c == speaker), alpha=0.5, label=f"speaker_{speaker}")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.8])
        ax.legend(loc="upper center", ncol=idx+1, bbox_to_anchor=(0.5, -0.1))
        plt.show()
        
        
class PyTorchPredictor(BasePredictor):
    def __init__(self, config_path, model_path, max_frame=45, hop=3):
        super().__init__(config_path, max_frame, hop)
        
        weight = torch.load(model_path, map_location="cpu")
        self.model = Encoder(self.ndim).to(self.device)
        self.model.load_state_dict(weight)
        self.model.eval()
    
    def predict(self, path, plot=False):        
        y = self._load(path, mfcc=False)
        activity = zcr_vad(y)
        spans = get_timestamp(activity)
        
        embed = [self._encode_segment(y, span) for span in spans]
        embed = torch.cat(embed).cpu().numpy()
        speakers = OptimizedAgglomerativeClustering().fit_predict(embed)
        
        if plot:
            self._plot_diarization(y, spans, speakers)
            
        timestamp = np.array(spans) / self.sr
        return timestamp, speakers
    
    def _encode_segment(self, y, span):
        start, end = span
        mfcc = self._mfcc(y[:, start:end]).to(self.device)
        mfcc = mfcc.unfold(2, self.max_frame, self.hop).permute(2, 0, 1, 3)
        with torch.no_grad():
            embed = self.model(mfcc).mean(0, keepdims=True)
        return embed
        
    def to_onnx(self, fname="speaker_diarization.onnx", outdir="model/openvino"):
        os.makedirs(outdir, exist_ok=True)
        mfcc = torch.rand(1, 1, self.n_mfcc, self.max_frame).to(self.device)
        onnx.export(self.model, mfcc, f"{outdir}/{fname}", input_names=["input"], output_names=["output"])
        print(f"model is exported as {outdir}/{fname}")     
        

class OpenVINOPredictor(BasePredictor):
    def __init__(self, model_xml, model_bin, config_path, max_frame=45, hop=3):
        super().__init__(config_path, max_frame, hop)
        net = IENetwork(model_xml, model_bin)
        assert max_frame == net.inputs["input"].shape[-1]
        
        plugin = IECore() #
        self.exec_net = plugin.load_network(net, "CPU") #import network to CPU or GPU etc..

    def predict(self, path, plot=False):        
        y = self._load(path, mfcc=False) #load wav & change 44100 to 16000
        activity = zcr_vad(y)  # false
        spans = get_timestamp(activity) #調整維度       
        
        embed = [self._encode_segment(y, span) for span in spans]
        embed = np.vstack(embed) # 沿着竖直方向将矩阵堆叠起来
        speakers = OptimizedAgglomerativeClustering().fit_predict(embed) #建立說話者位置

        if plot:
            self._plot_diarization(y, spans, speakers)  #print picture
            
        timestamp = np.array(spans) / self.sr
        print(timestamp, "timestamp==========================>")
        
        oldfoldname = "audiocaut/"
        newfoldname = "finalaudio/"
        if os.path.exists(oldfoldname):
            shutil.rmtree(oldfoldname)
            os.mkdir(oldfoldname)
        else:
            os.mkdir(oldfoldname)
            
        newaud = newfoldname + "new.wav"
        dataDir = newaud
        targetDir = oldfoldname
        
        for cutsec in timestamp:
            print(cutsec[0], cutsec[1], "wav time==============>")
            
            get_wav_make(dataDir, targetDir, cutsec[0], cutsec[1])
            
            print(cutsec, cutsec, "time=============================>\n")
                
        return timestamp, speakers
    
    def _encode_segment(self, y, span):
        start, end = span
        mfcc = self._mfcc(y[:, start:end])
        
        print(mfcc.shape, "mfccshape=============>")
        mfcc = mfcc.unfold(2, self.max_frame, self.hop).permute(2, 0, 1, 3)
        print(mfcc.shape, "mfcc unflod==================>\n")
        mfcc = mfcc.cpu().numpy()
        
        embed = [self.exec_net.infer({"input": m}) for m in mfcc]
        embed = np.array([e["output"] for e in embed])
        embed = embed.mean(0)
        return embed
