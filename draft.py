import pyaudio
import wave
import struct
import math
import numpy as np
import scipy.signal as sig
import matplotlib
import tkinter as Tk
import threading
import queue
import time

from enum import Enum, auto
from dataclasses import dataclass, field
from tkinter import ttk
from math import sin, cos, pi
from math import e
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class GUIState(Enum):
    NORMAL_OP           = 0
    EDIT_START_POS      = auto()
    EDIT_STOP_POS       = auto()


class MsgType(Enum):
    UNSPECIFIED         = 0
    LOAD_AUDIO          = auto()
    RECORD_AUDIO        = auto()
    PLAY_FULL_SOURCE    = auto()
    STOP_FULL_CLIP      = auto()
    PLAY_SLICE_1        = auto()
    PLAY_SLICE_2        = auto()
    PLAY_SLICE_3        = auto()
    PLAY_SLICE_4        = auto()
    PLAY_SLICE_5        = auto()
    PLAY_SLICE_6        = auto()
    PLAY_SLICE_7        = auto()
    PLAY_SLICE_8        = auto()
    PLAY_SLICE_9        = auto()
    PLAY_SLICE_10       = auto()
    PLAY_SLICE_11       = auto()
    PLAY_SLICE_12       = auto()
    PLAY_SLICE_13       = auto()
    PLAY_SLICE_14       = auto()
    PLAY_SLICE_15       = auto()
    PLAY_SLICE_16       = auto()
    CREATE_SLICE        = auto()
    


    
# Shared memory for audio clips
class AudioClip():
    def __init__(self, signal=None):
        self.playing = False    # Is clip currently playing?
        self.idx = 0            # Current index to play from
        self.signal = signal    # Signal data 

    def start_at_index(self, idx):
        self.idx = idx
        return self
    
    def cur_block(self):
        block = self.signal[self.idx]
        return block
    
    def get_block(self, blocklen):
        endblock = self.idx + blocklen
        block = self.signal[self.idx:endblock]
        self.idx = endblock
        return block
    
    def clip_ended(self, blocklen):
        if self.idx >= len(self.signal)-blocklen:
            return True
        else:
            return False


@dataclass
class SliceParams():
    start_idx: int      = 0
    stop_idx: int       = 0
    created: bool       = False



# Shared memory for GUI data
@dataclass
class GUIData():
    low_pass_freq: int                  = 0
    audio_filename: str                 = None
    cur_slice: int                      = 0
    pa_format: int                      = pyaudio.paInt16
    sample_num_channels: int            = 1
    sample_num_frames: int              = 0
    sample_width: int                   = 2
    sample_rate: int                    = 8000
    blocklen: int                       = 64
    gui_state: GUIState                 = GUIState.NORMAL_OP
    # slice_params: list[SliceParams]     = [SliceParams() for i in range(16)]
    slice_params: list[SliceParams]     = field(default_factory=list)





class WorkspaceGUI(Tk.Tk):
    def __init__ (self, *args, **kwargs):
        Tk.Tk.__init__(self, *args, **kwargs)

        self.queue = queue.Queue()
        self.shm = GUIData()
        self.shm.slice_params = [SliceParams() for i in range(16)]

        container = Tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)

        vis_frame = VisFrame(container, self.queue, self.shm, width=1500, height=500)
        vis_frame.grid(row=0, column=0, padx=10, pady=(10,5))

        mod_frame = ModFrame(container, self.queue, self.shm, width=1500, height=500)
        mod_frame.grid(row=1, column=0, padx=10, pady=(5,10))

        self.i = 0

        self.abe = AudioBackend(self.queue, self.shm)
        self.abe.start()



        print("GUI: {}".format(threading.current_thread().name))

        self.after(200, self.check_queue)

    def test_func(self):
        print("Test func: {}".format(self.i))
        self.i += 1
        time.sleep(0.5)

    def check_queue(self):
        # try:
        #     while self.queue.qsize() > 0:
        #         item = self.queue.get(block=False)
        #         print("Got from queue: {}".format(item))
        # except queue.Empty:
        #     pass

        # Handle anything from the queue (here or in the 'try' above)

        self.after(10, self.check_queue)

    def shutdown(self):
        print("shutting down")
        self.abe.stop()
        self.abe.join()
        self.quit()





class VisFrame(Tk.Frame):
    def __init__(self, parent, queue:queue.Queue, shm:GUIData, *args, **kwargs):
        Tk.Frame.__init__(self, parent, *args, **kwargs)

        self.queue = queue
        self.shm = shm

        self.create_clear_figure()
        self.create_frame()


    def create_clear_figure(self):
        self.fig = plt.figure(1, figsize=(5, 4))
        self.ax1 = self.fig.add_subplot(1,1,1)
        [self.g1] = self.ax1.plot([],[])

        self.ax1.set_xlim(0, 22000)
        self.ax1.set_ylim(-(2**(8*2)/2), (2**(8*2)/2)-1)
        self.ax1.set_xlabel('Time (msec)')
        self.ax1.set_title("Input Signal")
        self.ax1.get_yaxis().set_visible(False)

        x = np.arange(1000)
        y = [0] * 1000

        self.g1.set_xdata(x)
        self.g1.set_ydata(y)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.fig.tight_layout()

        self.fig_canvas_agg = FigureCanvasTkAgg(self.fig, master=self)

    
    def on_plot_click(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        
        if self.shm.gui_state == GUIState.EDIT_START_POS:
            # x_idx = event.xdata * self.shm.sample_rate / 1000
            x_idx = event.xdata
            self.shm.slice_params[self.shm.cur_slice].start_idx = int(x_idx)
            
            self.shm.gui_state = GUIState.EDIT_STOP_POS

        elif self.shm.gui_state == GUIState.EDIT_STOP_POS:
            # x_idx = event.xdata * self.shm.sample_rate / 1000
            x_idx = event.xdata
            self.shm.slice_params[self.shm.cur_slice].stop_idx = int(x_idx)
            self.shm.slice_params[self.shm.cur_slice].created = True

            self.queue.put(MsgType.CREATE_SLICE)

            self.shm.gui_state = GUIState.NORMAL_OP
        
        # x_idx = event.xdata * self.sample_rate / 1000
        
        # self.create_slice(x_idx)

    def create_frame(self):
        self.fig_canvas = self.fig_canvas_agg.get_tk_widget()
        self.fig_canvas.pack(side=Tk.LEFT)


    




class ModFrame(Tk.Frame):
    def __init__(self, parent:Tk.Frame, queue:queue.Queue, shm:GUIData, *args, **kwargs):
        Tk.Frame.__init__(self, parent, *args, **kwargs)

        self.root: WorkspaceGUI = parent.master

        self.queue = queue
        self.shm = shm

        self.create_frame()
        

    def create_frame(self):
        self.play_btn = ttk.Button(master=self, text='play', command=lambda:self.send_event(MsgType.PLAY_FULL_SOURCE))
        self.play_btn.pack()

        self.edit_start_btn = ttk.Button(master=self, text='edit start', command=lambda:self.set_gui_state(GUIState.EDIT_START_POS))
        self.edit_start_btn.pack()

        self.btn_1 = ttk.Button(master=self, text='1', command=lambda:self.select_slice(0))
        self.btn_1.pack()

        self.quit_btn = ttk.Button(master=self, text='quit', command=self.root.shutdown)
        self.quit_btn.pack()


    def send_event(self, event:MsgType):
        self.queue.put(event)

    
    def set_gui_state(self, state:GUIState):
        self.shm.gui_state = state

    
    def select_slice(self, slice_num:int):
        self.shm.cur_slice = slice_num
        self.queue.put(MsgType.PLAY_SLICE_1)


    def quit(self):
        self.root.shutdown()


    def null_test(self):
        pass






class AudioBackend(threading.Thread):
    def __init__(self, queue:queue.Queue, shm:GUIData):
        threading.Thread.__init__(self, name='audio_thread')
        self.queue = queue
        self.shm = shm


    def init_params(self):
        self.shutdown_event = threading.Event()

        self.blocklen = self.shm.blocklen

        # Audio params
        self.pa_format = pyaudio.paInt16
        self.sample_num_channels = 1
        self.sample_rate = 8000
        self.blocklen = 64

        self.all_clips: list[AudioClip] = []
        self.slices: list[AudioClip] = [AudioClip() for i in range(16)]

        self.full_source = AudioClip()
        self.all_clips.append(self.full_source)
        self.all_clips + self.slices

        self.audio_to_play = False
        self.clips_to_play: list[AudioClip] = []

        self.clean_output()

        # Testing
        self.sample_file_name = "author.wav"
        self.load_audio_file(self.sample_file_name)

        self.create_playback_stream()


    def run(self):
        itr = 0
        print("audio backend: {}".format(threading.current_thread().name))
        self.init_params()
        while True:
            if self.shutdown_event.is_set():
                print("Shutting down audio thread")
                break

            self.check_queue()
            self.play_audio()


    def stop(self):
        self.shutdown_event.set()


    def check_queue(self):
        try:
            while self.queue.qsize() > 0:
                event = self.queue.get(block=False)
                print("Got event from queue: {}".format(event))

                if event is MsgType.PLAY_FULL_SOURCE:
                    if not self.full_source.playing:
                        print("Playing audio")
                        self.stop_all_signals()
                        self.play_clip_from_idx_0(self.full_source.start_at_index(0))

                if event is MsgType.PLAY_SLICE_1:
                    # self.play_clip_from_idx_0(self.slices[self.shm.cur_slice].start_at_index(0))
                    self.play_slice_from_idx_0(self.shm.cur_slice)

                if event is MsgType.CREATE_SLICE:
                    self.create_slice(self.shm.cur_slice)
                    
        except queue.Empty:
            pass


    def clean_output(self):
        self.output_signal = np.zeros(self.shm.blocklen)


    def stop_all_signals(self):
        self.audio_to_play = False
        del self.clips_to_play[:]

        for clip in self.all_clips:
            clip.idx = 0

    
    def create_slice(self, slice_idx):
        start = self.shm.slice_params[slice_idx].start_idx
        stop = self.shm.slice_params[slice_idx].stop_idx
        sig = self.full_source.signal[start:stop]
        self.slices[slice_idx].signal = sig



    def play_clip_from_idx_0(self, clip: AudioClip):
        self.audio_to_play = True
        self.clips_to_play.append(clip)


    def play_slice_from_idx_0(self, slice_idx: int):
        self.slices[slice_idx].playing = True
        self.slices[slice_idx].idx = 0


    def create_output_signal(self):
        # Add given clip to output signal, starting with clip's current index
        for clip in self.slices:
            if clip.playing:
                if clip.clip_ended(self.shm.blocklen):
                    # If clip ended, remove from list
                    clip.playing = False
                    # self.clips_to_play.remove(clip)
                    continue
                
                self.output_signal += clip.get_block(self.shm.blocklen)


    def play_audio(self):
        # Play output_signal
        audio_to_play = False

        for clip in self.slices:
            if clip.playing == True:
                audio_to_play = True

        # if self.clips_to_play:
        if audio_to_play:
            self.create_output_signal()
            # print("output signal: {}".format(self.output_signal))
            out = self.output_signal.astype('int16').tobytes()
            self.stream.write(out)
            self.clean_output()


    def load_audio_file(self, filename):
        self.sample = wave.open(filename, 'rb')

        self.shm.sample_num_channels = self.sample.getnchannels()
        self.shm.sample_rate = self.sample.getframerate()
        self.shm.sample_num_frames = self.sample.getnframes()
        self.shm.sample_width = self.sample.getsampwidth()

        # signal_bytes = self.sample.readframes(1)

        if self.shm.sample_width == 1:
            format_opt = 'B'
        elif self.shm.sample_width == 2:
            format_opt = 'h'
        elif self.shm.sample_width == 4:
            format_opt = 'i'
        else:
            print("ERROR: unrecognized sample width {} - should be 1, 2, or 4")


        self.full_source = AudioClip()

        frames = self.sample.readframes(self.shm.sample_num_frames * self.shm.sample_num_channels)
        self.full_source.signal = struct.unpack_from(str(self.shm.sample_num_frames * self.shm.sample_num_channels) + format_opt, frames)

        if self.shm.sample_num_channels == 2:
            self.left_chan = [self.full_source.signal[i] for i in range(0, len(self.full_source.signal), 2)]
            self.right_chan = [self.full_source.signal[i] for i in range(1, len(self.full_source.signal), 2)]

        self.full_source.signal = np.asarray(self.full_source.signal)


        print("Full source shape: {}".format(self.full_source.signal.shape))


    def record_stream(self, stream=None):
        pass


    def create_playback_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.shm.pa_format,
            channels=self.shm.sample_num_channels,
            rate=self.shm.sample_rate,
            input=False,
            output=True,
            frames_per_buffer=self.shm.blocklen
        )




























class Workspace():
    def __init__(self, parent_instance=None):
        # Tkinter stuff
        self.root = parent_instance
        
        # Audio params
        self.pa_format = pyaudio.paInt16
        self.sample_num_channels = 1
        self.sample_rate = 8000
        self.blocklen = 64



        # Testing
        self.sample_file_name = "author.wav"
        self.t = np.arange(self.blocklen)

        self.load_audio_file(self.sample_file_name)
        self.t = np.arange(self.sample_num_frames) * 1000/self.sample_rate  # x axis in ms
        print("t shape: {}".format(self.t.shape))

        self.slice_markers = []

        self.create_figure()
        self.create_window()
        self.animate_plot()

        self.create_playback_stream()



    def load_audio_file(self, filename):
        self.sample = wave.open(filename, 'rb')

        self.sample_num_channels = self.sample.getnchannels()
        self.sample_rate = self.sample.getframerate()
        self.sample_num_frames = self.sample.getnframes()
        self.sample_width = self.sample.getsampwidth()

        # signal_bytes = self.sample.readframes(1)

        if self.sample_width == 1:
            format_opt = 'B'
        elif self.sample_width == 2:
            format_opt = 'h'
        elif self.sample_width == 4:
            format_opt = 'i'
        else:
            print("ERROR: unrecognized sample width {} - should be 1, 2, or 4")


        frames = self.sample.readframes(self.sample_num_frames * self.sample_num_channels)
        self.raw_signal = struct.unpack_from(str(self.sample_num_frames * self.sample_num_channels) + format_opt, frames)

        if self.sample_num_channels == 2:
            self.left_chan = [self.raw_signal[i] for i in range(0, len(self.raw_signal), 2)]
            self.right_chan = [self.raw_signal[i] for i in range(1, len(self.raw_signal), 2)]

        self.raw_signal = np.asarray(self.raw_signal)

        print("raw signal shape: {}".format(self.raw_signal.shape))


    def record_stream(self, stream=None):
        pass


    def create_playback_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.pa_format,
            channels=self.sample_num_channels,
            rate=self.sample_rate,
            input=False,
            output=True,
            frames_per_buffer=self.blocklen
        )


    def play_entire_sample(self):
        print("raw_sig shape: {}".format(self.raw_signal.shape))
        output_sig = self.raw_signal.astype('int16').tobytes()
        self.stream.write(output_sig)

    
    def play_from_idx(self, idx):
        print("idx: {}".format(idx))
        output_sig = self.raw_signal[idx:]
        print("play_from_idx - output_sig: {} // raw_signal: {}".format(output_sig.shape, self.raw_signal.shape))
        output_sig = output_sig.astype('int16').tobytes()
        self.stream.write(output_sig)


    def create_slice(self, xpos):
        xpos = int(xpos)
        if xpos not in self.slice_markers:
            self.slice_markers.append(xpos)

        print("slice markers: {}".format(self.slice_markers))


    def btn_1_func(self):
        self.play_slice(self.slice_markers[0])
        

    def play_slice(self, x_start):
        self.play_from_idx(x_start)


    def create_figure(self):
        self.fig = plt.figure(1)
        self.ax1 = self.fig.add_subplot(1,1,1)
        [self.g1] = self.ax1.plot([],[])
        self.ax1.set_xlim(0, (self.sample_num_frames / self.sample_rate) * 1000)
        self.ax1.set_ylim(-(2**(8*self.sample_width)/2), (2**(8*self.sample_width)/2)-1)
        self.ax1.set_xlabel('Time (msec)')
        self.ax1.set_title("Input Signal")
        self.ax1.get_yaxis().set_visible(False)

        self.g1.set_xdata(self.t)
        self.g1.set_ydata(self.raw_signal)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.fig.tight_layout()

        self.fig_canvas_agg = FigureCanvasTkAgg(self.fig, master=self.root)

    
    def on_plot_click(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        
        x_idx = event.xdata * self.sample_rate / 1000
        self.create_slice(x_idx)


    def create_window(self):
        self.fig_canvas = self.fig_canvas_agg.get_tk_widget()
        self.fig_canvas.pack()

        self.play_btn = ttk.Button(master=self.root, text='play', command=self.play_entire_sample)
        self.play_btn.pack()

        self.btn_1 = ttk.Button(master=self.root, text='1', command=self.btn_1_func)
        self.btn_1.pack()

        self.quit_btn = ttk.Button(master=self.root, text='quit', command=self.root.quit)
        self.quit_btn.pack()




    def update_animation(self, i):
        self.g1.set_ydata(self.raw_signal)
        return (self.g1,)


    def init_signal_plots(self):
        self.g1.set_xdata(self.t)
        return (self.g1,)
    

    def animate_plot(self):
        self.animate = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            init_func=self.init_signal_plots,
            interval=10,
            blit=True,
            repeat=False
        )
        




if __name__ == "__main__":
    matplotlib.use('TkAgg')

    # root = Tk.Tk()
    # ws = Workspace(parent_instance=root)
    # root.mainloop()





    app = WorkspaceGUI()
    app.mainloop()
