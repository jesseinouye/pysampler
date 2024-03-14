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
from tkinter import ttk, filedialog
from math import sin, cos, pi
from math import e
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



keymap = {
    "a":    0,
    "w":    1,
    "s":    2,
    "e":    3,
    "d":    4,
    "f":    5,
    "t":    6,
    "g":    7,
    "y":    8,
    "h":    9,
    "u":    10,
    "j":    11,
    "k":    12,
    "o":    13,
    "l":    14,
    "p":    15,
    ";":    16,
    "'":    17
}


class GUIState(Enum):
    NORMAL_OP           = 0
    EDIT_START_POS      = auto()
    EDIT_STOP_POS       = auto()


class MsgType(Enum):
    UNSPECIFIED         = 0
    LOAD_AUDIO          = auto()
    RECORD_AUDIO_INPUT  = auto()
    STOP_REC_INPUT      = auto()
    UPDATE_PLOT         = auto()
    PLAY_FULL_SOURCE    = auto()
    STOP_FULL_CLIP      = auto()
    INPUT_STREAM_START  = auto()
    RECORD_OUTPUT       = auto()
    STOP_REC_OUTPUT     = auto()
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
        self.mono_signal = None

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
    apply_lowpass: bool                 = False
    lowpass_freq: int                   = 500
    apply_highpass: bool                = False
    highpass_freq: int                  = 500
    apply_bandpass: bool                = False
    bandpass_low_freq: int              = 500
    bandpass_high_freq: int             = 2000
    apply_am: bool                      = False
    am_mod_freq: int                    = 200
    audio_filename: str                 = None
    cur_slice: int                      = 0
    pa_format: int                      = pyaudio.paInt16
    sample_num_channels: int            = 1
    sample_num_frames: int              = 0
    sample_width: int                   = 2
    sample_rate: int                    = 16000
    blocklen: int                       = 256
    gui_state: GUIState                 = GUIState.NORMAL_OP
    # slice_params: list[SliceParams]     = [SliceParams() for i in range(16)]
    slice_params: list[SliceParams]     = field(default_factory=list)
    clips: list[AudioClip]              = field(default_factory=list)
    audio_loaded: bool                  = False
    recording_input: bool               = False
    gain: float                         = 1.0







class WorkspaceGUI(Tk.Tk):
    def __init__ (self, *args, **kwargs):
        Tk.Tk.__init__(self, *args, **kwargs)

        self.audio_queue = queue.Queue()
        self.gui_queue = queue.Queue()
        self.shm = GUIData()
        self.shm.slice_params = [SliceParams() for i in range(16)]

        self.abe = AudioBackEnd(self.audio_queue, self.gui_queue, self.shm)
        self.abe.start()

        self.container = Tk.Frame(self)
        self.container.pack(side='top', fill='both', expand=True)

        self.vis_frame = VisFrame(self.container, self.audio_queue, self.shm, width=1500, height=500)
        self.vis_frame.grid(row=0, column=0, padx=10, pady=(10,5))

        self.mod_frame = ModFrame(self.container, self.audio_queue, self.shm, width=1500, height=500)
        self.mod_frame.grid(row=1, column=0, padx=10, pady=(5,10))

        self.grid_rowconfigure((0,1), weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.i = 0

        self.bind("<Key>", self.key_press)

        self.keymap = keymap

        print("GUI: {}".format(threading.current_thread().name))

        self.after(200, self.check_queue)

    def test_func(self):
        print("Test func: {}".format(self.i))
        self.i += 1
        time.sleep(0.5)

    def key_press(self, event:Tk.Event):
        if event.char in self.keymap:
            self.shm.cur_slice = self.keymap[event.char]
            # self.mod_frame.pad_buttons[self.shm.cur_slice].config(default='active')
            self.audio_queue.put(MsgType.PLAY_SLICE_1)
            # self.mod_frame.pad_buttons[self.shm.cur_slice].config(default='active')
            self.mod_frame.info_var.set("Slice: {}".format(self.keymap[event.char]))


    def check_queue(self):
        # try:
        #     while self.queue.qsize() > 0:
        #         item = self.queue.get(block=False)
        #         print("Got from queue: {}".format(item))
        # except queue.Empty:
        #     pass

        # Handle anything from the queue (here or in the 'try' above)
        try:
            while self.gui_queue.qsize() > 0:
                event = self.gui_queue.get(block=False)
                # print("Got event from queue: {}".format(event))

                if event is MsgType.UPDATE_PLOT:
                    self.vis_frame.create_audio_figure()
                    self.mod_frame.info_var.set("Audio loaded")

                if event is MsgType.INPUT_STREAM_START:
                    self.mod_frame.info_var.set("Stream started - recording input")

        except queue.Empty:
            pass
        

        self.after(10, self.check_queue)

    def shutdown(self):
        print("shutting down")
        self.abe.stop()
        self.abe.join()
        self.quit()





class VisFrame(Tk.Frame):
    def __init__(self, parent:Tk.Frame, queue:queue.Queue, shm:GUIData, *args, **kwargs):
        Tk.Frame.__init__(self, parent, *args, **kwargs)

        self.queue = queue
        self.shm = shm

        self.create_clear_figure()
        self.create_frame()

    def create_clear_figure(self):
        self.fig = plt.figure(1, figsize=(10, 4))
        self.ax1 = self.fig.add_subplot(1,1,1)
        [self.g1] = self.ax1.plot([],[])

        self.ax1.set_xlim(0, (80000 / 8000) * 1000)
        self.ax1.set_ylim(-(2**(8*1)/2), (2**(8*1)/2)-1)

        # self.ax1.set_xlim(0, 22000)
        # self.ax1.set_ylim(-(2**(8*2)/2), (2**(8*2)/2)-1)
        self.ax1.set_xlabel('Time (msec)')
        self.ax1.set_title("Input Signal")
        self.ax1.get_yaxis().set_visible(False)

        # x = np.arange(1000)
        # y = [0] * 1000

        x = np.arange(80000) * 1000/8000
        y = [0] * len(x)

        self.g1.set_xdata(x)
        self.g1.set_ydata(y)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.fig.tight_layout()

        self.fig_canvas_agg = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = NavigationToolbar2Tk(self.fig_canvas_agg, self)
        self.toolbar.update()

        plt.rcParams['keymap.save'].remove('s')
        plt.rcParams['keymap.fullscreen'].remove('f')
        plt.rcParams['keymap.pan'].remove('p')
        plt.rcParams['keymap.zoom'].remove('o')
        plt.rcParams['keymap.grid'].remove('g')
        plt.rcParams['keymap.yscale'].remove('l')
        plt.rcParams['keymap.xscale'].remove('k')
        

    def create_audio_figure(self):
        self.fig.clear()

        self.ax1 = self.fig.add_subplot(1,1,1)
        [self.g1] = self.ax1.plot([],[])

        self.ax1.set_xlim(0, (self.shm.sample_num_frames / self.shm.sample_rate) * 1000)
        self.ax1.set_ylim(-(2**(8*self.shm.sample_width)/2), (2**(8*self.shm.sample_width)/2)-1)

        self.ax1.set_xlabel('Time (msec)')
        self.ax1.set_title("Input Signal")
        self.ax1.get_yaxis().set_visible(False)

        x = np.arange(self.shm.sample_num_frames) * 1000/self.shm.sample_rate
        
        if self.shm.sample_num_channels == 1:
            y = self.shm.clips[0].signal
        else:
            y = self.shm.clips[0].mono_signal

        self.g1.set_xdata(x)
        # self.g1.set_ydata(self.shm.clips[0].signal)
        self.g1.set_ydata(y)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.fig.tight_layout()

        self.fig_canvas_agg.draw_idle()


    def get_time_from_index(self, idx):
        return (idx / self.shm.sample_rate) * 1000

    
    def on_plot_click(self, event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   ('double' if event.dblclick else 'single', event.button,
        #    event.x, event.y, event.xdata, event.ydata))
        
        if self.shm.gui_state == GUIState.EDIT_START_POS:
            x_idx = event.xdata * self.shm.sample_rate / 1000
            # x_idx = event.xdata
            print("Start time: {}".format(self.get_time_from_index(x_idx)))
            self.shm.slice_params[self.shm.cur_slice].start_idx = int(x_idx)

            self.master.master.mod_frame.info_var.set("Click plot to select end of slice.")
            
            self.shm.gui_state = GUIState.EDIT_STOP_POS

        elif self.shm.gui_state == GUIState.EDIT_STOP_POS:
            x_idx = event.xdata * self.shm.sample_rate / 1000
            # x_idx = event.xdata
            print("Stop time: {}".format(self.get_time_from_index(x_idx)))
            self.shm.slice_params[self.shm.cur_slice].stop_idx = int(x_idx)
            self.shm.slice_params[self.shm.cur_slice].created = True

            self.queue.put(MsgType.CREATE_SLICE)

            self.master.master.mod_frame.info_var.set("Slice {} created!".format(self.shm.cur_slice))

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

        # self.shm.apply_lowpass = self.lowpass_var
        # self.shm.apply_highpass = self.highpass_var
        # self.shm.apply_bandpass = self.bandpass_var
        

    def create_frame(self):
        # Left frame for pad grid
        self.pad_frame = Tk.Frame(master=self, width=460, height=450)
        self.pad_frame.grid(column=0, row=0, sticky="w")
        self.pad_frame.pack_propagate(False)
        self.pad_frame.grid_propagate(False)

        self.pad_buttons = []

        self.pad_group = Tk.Frame(master=self.pad_frame)

        for x in range(4):
            for y in range(4):
                btn = Tk.Button(self.pad_group, text=str((x*4)+y), height=4, width=6, command=lambda x=x, y=y:self.select_slice((x*4)+y))
                btn.grid(column=y, row=x, sticky="nsew")
                self.pad_buttons.append(btn)
                

        self.pad_group.columnconfigure(tuple(range(4)), weight=1)
        self.pad_group.rowconfigure(tuple(range(4)), weight=1)

        self.pad_group.place(in_=self.pad_frame, anchor="c", relx=.5, rely=.5)


        # Right frame for control grid
        self.ctrl_frame = Tk.Frame(master=self, width=600, height=450)
        self.ctrl_frame.grid(column=1, row=0, sticky="w")
        self.ctrl_frame.pack_propagate(False)
        self.ctrl_frame.grid_propagate(False)

        self.audio_ctrl_frame = Tk.Frame(master=self.ctrl_frame)
        self.audio_ctrl_frame.grid(column=0, row=0)
        # self.audio_ctrl_frame.pack_propagate(False)
        # self.audio_ctrl_frame.grid_propagate(False)

        self.main_ctrl_frame = Tk.Frame(master=self.ctrl_frame)
        self.main_ctrl_frame.grid(column=1, row=0)
        # self.main_ctrl_frame.pack_propagate(False)
        # self.main_ctrl_frame.grid_propagate(False)


        # Audio control frame


        # self.play_btn = ttk.Button(master=self.audio_ctrl_frame, text='Play Track', command=lambda:self.send_event(MsgType.PLAY_FULL_SOURCE))
        # self.play_btn.grid(column=0, row=0, sticky='nsew')
        
        self.edit_start_btn = ttk.Button(master=self.audio_ctrl_frame, text='Create Slice', command=lambda:self.build_slice())
        self.edit_start_btn.grid(column=0, row=1, sticky='nsew')

        # Low-pass filter
        self.lowpass_label = ttk.Label(self.audio_ctrl_frame, text="Low-pass Filter")
        self.lowpass_label.grid(column=0, row=2, sticky='w')

        self.lowpass_var = Tk.IntVar()
        self.lowpass_check = ttk.Checkbutton(self.audio_ctrl_frame, variable=self.lowpass_var, onvalue=1, offvalue=0, command=self.update_lowpass)
        self.lowpass_check.grid(column=1, row=2, sticky='w', padx=(0, 10))

        self.lowpass_scale_var = Tk.IntVar()
        self.lowpass_scale = ttk.Scale(self.audio_ctrl_frame, from_=50, to=7850, length=250, orient='horizontal', command=self.update_lowpass_freq)
        self.lowpass_scale.set(1000)
        self.lowpass_scale.grid(column=2, row=2, sticky='w')

        self.lowpass_scale_var.set(int(self.lowpass_scale.get()))
        self.lowpass_num = Tk.Label(master=self.audio_ctrl_frame, textvariable=self.lowpass_scale_var, anchor='w', justify=Tk.LEFT, fg='#AAA')
        self.lowpass_num.grid(column=3, row=2, sticky='w')

        # High-pass filter
        self.highpass_label = ttk.Label(self.audio_ctrl_frame, text="High-pass Filter")
        self.highpass_label.grid(column=0, row=3, sticky='w')

        self.highpass_var = Tk.IntVar()
        self.highpass_check = ttk.Checkbutton(self.audio_ctrl_frame, variable=self.highpass_var, onvalue=1, offvalue=0, command=self.update_highpass)
        self.highpass_check.grid(column=1, row=3, sticky='w', padx=(0, 10))

        self.highpass_scale_var = Tk.IntVar()
        self.highpass_scale = ttk.Scale(self.audio_ctrl_frame, from_=50, to=7850, length=250, orient='horizontal', command=self.update_highpass_freq)
        self.highpass_scale.set(1000)
        self.highpass_scale.grid(column=2, row=3, sticky='w')

        self.highpass_scale_var.set(int(self.highpass_scale.get()))
        self.highpass_num = Tk.Label(master=self.audio_ctrl_frame, textvariable=self.highpass_scale_var, anchor='w', justify=Tk.LEFT, fg='#AAA')
        self.highpass_num.grid(column=3, row=3, sticky='w')


        # Band-pass filter
        self.bandpass_label = ttk.Label(self.audio_ctrl_frame, text="Band-pass Filter")
        self.bandpass_label.grid(column=0, row=4, sticky='w')

        self.bandpass_var = Tk.IntVar()
        self.bandpass_check = ttk.Checkbutton(self.audio_ctrl_frame, variable=self.bandpass_var, onvalue=1, offvalue=0, command=self.update_bandpass)
        self.bandpass_check.grid(column=1, row=4, sticky='w', padx=(0, 10))

        self.bandpass_low_scale_var = Tk.IntVar()
        self.bandpass_low_scale = ttk.Scale(self.audio_ctrl_frame, from_=50, to=7850, length=250, orient='horizontal', command=self.update_bandpass_low_freq)
        self.bandpass_low_scale.set(500)
        self.bandpass_low_scale.grid(column=2, row=4, sticky='w')

        self.bandpass_low_scale_var.set(int(self.bandpass_low_scale.get()))
        self.bandpass_low_num = Tk.Label(master=self.audio_ctrl_frame, textvariable=self.bandpass_low_scale_var, anchor='w', justify=Tk.LEFT, fg='#AAA')
        self.bandpass_low_num.grid(column=3, row=4, sticky='w')

        self.bandpass_high_scale_var = Tk.IntVar()
        self.bandpass_high_scale = ttk.Scale(self.audio_ctrl_frame, from_=50, to=7850, length=250, orient='horizontal', command=self.update_bandpass_high_freq)
        self.bandpass_high_scale.set(1500)
        self.bandpass_high_scale.grid(column=2, row=5, sticky='w')

        self.bandpass_high_scale_var.set(int(self.bandpass_high_scale.get()))
        self.bandpass_high_num = Tk.Label(master=self.audio_ctrl_frame, textvariable=self.bandpass_high_scale_var, anchor='w', justify=Tk.LEFT, fg='#AAA')
        self.bandpass_high_num.grid(column=3, row=5, sticky='w')


        # Amplitude modulation
        self.am_label = ttk.Label(self.audio_ctrl_frame, text="Amplitude Modulation")
        self.am_label.grid(column=0, row=6, sticky='w')

        self.am_var = Tk.IntVar()
        self.am_check = ttk.Checkbutton(self.audio_ctrl_frame, variable=self.am_var, onvalue=1, offvalue=0, command=self.update_am)
        self.am_check.grid(column=1, row=6, sticky='w', padx=(0, 10))

        self.am_scale_var = Tk.IntVar()
        self.am_scale = ttk.Scale(self.audio_ctrl_frame, from_=-300, to=7850, length=250, orient='horizontal', command=self.update_am_freq)
        self.am_scale.set(0)
        self.am_scale.grid(column=2, row=6, sticky='w')

        self.am_scale_var.set(int(self.am_scale.get()))
        self.am_num = Tk.Label(master=self.audio_ctrl_frame, textvariable=self.am_scale_var, anchor='w', justify=Tk.LEFT, fg='#AAA')
        self.am_num.grid(column=3, row=6, sticky='w')


        # Gain
        self.gain_label = ttk.Label(self.audio_ctrl_frame, text="Gain")
        self.gain_label.grid(column=0, row=7, sticky='w')

        # self.am_var = Tk.IntVar()
        # self.am_check = ttk.Checkbutton(self.audio_ctrl_frame, variable=self.am_var, onvalue=1, offvalue=0, command=self.update_am)
        # self.am_check.grid(column=1, row=6, sticky='w', padx=(0, 10))

        self.gain_scale_var = Tk.IntVar()
        self.gain_scale = ttk.Scale(self.audio_ctrl_frame, from_=1, to=3, length=250, orient='horizontal', command=self.update_gain)
        self.gain_scale.set(0)
        self.gain_scale.grid(column=2, row=7, sticky='w')

        self.gain_scale_var.set(int(self.gain_scale.get()))
        self.gain_num = Tk.Label(master=self.audio_ctrl_frame, textvariable=self.gain_scale_var, anchor='w', justify=Tk.LEFT, fg='#AAA')
        self.gain_num.grid(column=3, row=7, sticky='w', )


        self.audio_ctrl_frame.rowconfigure((0,1), weight=1, minsize=20)
        self.audio_ctrl_frame.rowconfigure((0,1,2,3,4,5,6), minsize=50)


        # Main control frame


        self.load_audio_btn = ttk.Button(master=self.main_ctrl_frame, text='Load Audio', command=self.load_file)
        self.load_audio_btn.grid(column=0, row=0, sticky='nsew')

        self.record_input_btn = ttk.Button(master=self.main_ctrl_frame, text='Record Input', command=self.record_input)
        self.record_input_btn.grid(column=0, row=1, sticky='nsew')

        self.stop_input_btn = ttk.Button(master=self.main_ctrl_frame, text='Stop Input', command=self.stop_recording_input)
        self.stop_input_btn.grid(column=0, row=2, sticky='nsew')

        self.record_output_btn = ttk.Button(master=self.main_ctrl_frame, text='Record Output', command=self.record_output)
        self.record_output_btn.grid(column=0, row=3, sticky='nsew')

        self.stop_output_btn = ttk.Button(master=self.main_ctrl_frame, text='Stop Output', command=self.stop_recording_output)
        self.stop_output_btn.grid(column=0, row=4, sticky='nsew')

        self.quit_btn = ttk.Button(master=self.main_ctrl_frame, text='Quit', command=self.root.shutdown)
        self.quit_btn.grid(column=0, row=5, sticky='nsew')

        self.main_ctrl_frame.rowconfigure((0,1), weight=1, minsize=20)



        self.ctrl_frame.columnconfigure((0,1), weight=1)



        # Bottom frame for text info
        self.text_frame = Tk.Frame(master=self, width=1100, height=25)
        self.text_frame.grid(column=0, row=1, columnspan=2)
        self.text_frame.pack_propagate(False)
        self.text_frame.grid_propagate(False)

        self.info_var = Tk.StringVar()
        self.info_var.set("Welcome! Relevant usage info will appear here.")

        self.info_box = Tk.Label(master=self.text_frame, textvariable=self.info_var, anchor='w', justify=Tk.LEFT, fg='#888')
        self.info_box.pack(anchor='w')


    def update_lowpass(self):
        self.shm.apply_lowpass = self.lowpass_var.get()

    def update_lowpass_freq(self, f):
        self.shm.lowpass_freq = int(float(f))
        self.lowpass_scale_var.set(int(float(f)))

    def update_highpass(self):
        self.shm.apply_highpass = self.highpass_var.get()

    def update_highpass_freq(self, f):
        self.shm.highpass_freq = int(float(f))
        self.highpass_scale_var.set(int(float(f)))

    def update_bandpass(self):
        self.shm.apply_bandpass = self.bandpass_var.get()

    def update_bandpass_low_freq(self, f):
        self.shm.bandpass_low_freq = int(float(f))
        self.bandpass_low_scale_var.set(int(float(f)))

    def update_bandpass_high_freq(self, f):
        self.shm.bandpass_high_freq = int(float(f))
        self.bandpass_high_scale_var.set(int(float(f)))

    def update_am(self):
        self.shm.apply_am = self.am_var.get()

    def update_am_freq(self, f):
        self.shm.am_mod_freq = int(float(f))
        self.am_scale_var.set(int(float(f)))

    def update_gain(self, g):
        self.shm.gain = round(float(g), 3)
        self.gain_scale_var.set(round(float(g), 3))

    def send_event(self, event:MsgType):
        self.queue.put(event)

    def build_slice(self):
        self.info_var.set("Click plot to select beginning of slice.")
        self.set_gui_state(GUIState.EDIT_START_POS)


    def set_gui_state(self, state:GUIState):
        self.shm.gui_state = state

    
    def select_slice(self, slice_num:int):
        print("slice: {}".format(slice_num))
        self.shm.cur_slice = slice_num
        self.queue.put(MsgType.PLAY_SLICE_1)


    def load_file(self):
        self.info_var.set("Loading audio...")
        audio_path = filedialog.askopenfilename()

        if (audio_path != '') and (audio_path is not None):
            self.shm.audio_filename = audio_path
            self.send_event(MsgType.LOAD_AUDIO)
        else:
            print("No audio file provided, not loading")

    def record_input(self):
        self.info_var.set("Starting input stream...")

        self.shm.recording_input = True
        self.send_event(MsgType.RECORD_AUDIO_INPUT)

    def stop_recording_input(self):
        self.info_var.set("Stopping input recording...")
        self.shm.recording_input = False
        self.send_event(MsgType.STOP_REC_INPUT)

    def record_output(self):
        self.info_var.set("Recording output to wav file")
        self.send_event(MsgType.RECORD_OUTPUT)

    def stop_recording_output(self):
        self.info_var.set("Output recording stopped")
        self.send_event(MsgType.STOP_REC_OUTPUT)

    def quit(self):
        self.root.shutdown()


    def null_test(self):
        pass






class AudioBackEnd(threading.Thread):
    def __init__(self, audio_queue:queue.Queue, gui_queue:queue.Queue, shm:GUIData):
        threading.Thread.__init__(self, name='audio_thread')
        self.queue = audio_queue
        self.gui_queue = gui_queue
        self.shm = shm


    def init_params(self):
        self.shutdown_event = threading.Event()

        self.blocklen = self.shm.blocklen

        # Audio params
        self.pa_format = pyaudio.paInt16
        self.sample_num_channels = self.shm.sample_num_channels
        self.sample_rate = self.shm.sample_rate
        # self.blocklen = 256

        self.all_clips: list[AudioClip] = []
        self.slices: list[AudioClip] = [AudioClip(np.zeros(self.shm.blocklen)) for i in range(16)]

        self.full_source = AudioClip()
        self.all_clips.append(self.full_source)
        self.all_clips + self.slices

        self.audio_to_play = False
        self.clips_to_play: list[AudioClip] = []

        self.theta = 0
        self.filter_order = 7
        self.bandpass_states = np.zeros(self.filter_order*2)
        self.highpass_states = np.zeros(self.filter_order)
        self.lowpass_states = np.zeros(self.filter_order)
        self.am_mod_states = np.zeros(self.filter_order)

        self.recording_input = False
        self.recording_output = False

        self.p = pyaudio.PyAudio()

        self.build_AM_params()

        self.stream = None
        self.create_playback_stream()

        self.clean_output()

        # Testing
        # self.sample_file_name = "author.wav"
        # self.sample_file_name = "rushing_back_left.wav"
        # self.load_audio_file(self.sample_file_name)

        # self.create_playback_stream()

        # self.b, self.a = self.create_bandpass_filter()


    def run(self):
        itr = 0
        print("audio backend: {}".format(threading.current_thread().name))
        self.init_params()
        while True:
            if self.shutdown_event.is_set():
                print("Shutting down audio thread")
                self.stream.stop_stream()
                self.stream.close()
                break

            self.check_queue()
            # self.play_audio()

            if self.recording_input:
                self.record_audio_input()
            else:
                self.play_audio()
            
            # if self.recording_output:
            #     self.record_audio_output()


    def stop(self):
        self.shutdown_event.set()


    def send_event(self, event:MsgType):
        self.gui_queue.put(event)


    def check_queue(self):
        try:
            while self.queue.qsize() > 0:
                event = self.queue.get(block=False)
                # print("Got event from queue: {}".format(event))

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

                if event is MsgType.LOAD_AUDIO:
                    self.load_audio_file(self.shm.audio_filename)

                if event is MsgType.RECORD_AUDIO_INPUT:
                    if not self.recording_input:
                        self.recording_input = True
                        self.init_record_audio_input()

                if event is MsgType.STOP_REC_INPUT:
                    if self.recording_input:
                        self.recording_input = False
                        self.save_recorded_audio_input()

                if event is MsgType.RECORD_OUTPUT:
                    if not self.recording_output:
                        self.recording_output = True
                        self.init_record_audio_output()

                if event is MsgType.STOP_REC_OUTPUT:
                    if self.recording_output:
                        self.recording_output = False
                        self.close_record_audio_out()

                    
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


    def create_bandpass_filter(self):
        # Wn = [200, 2000]
        Wn = [self.shm.bandpass_low_freq, self.shm.bandpass_high_freq]
        b, a = sig.butter(self.filter_order, Wn, btype='bandpass', fs=self.shm.sample_rate)
        return b, a
    

    def create_lowpass_filter(self):
        # Wn = 200
        Wn = self.shm.lowpass_freq
        b, a = sig.butter(self.filter_order, Wn, btype='lowpass', fs=self.shm.sample_rate)
        return b, a
    

    def create_highpass_filter(self):
        # Wn = 200
        Wn = self.shm.highpass_freq
        b, a = sig.butter(self.filter_order, Wn, btype='highpass', fs=self.shm.sample_rate)
        return b, a


    def apply_filters(self):
        if self.shm.apply_lowpass:
            b, a = self.create_lowpass_filter()
            [o, states] = sig.lfilter(b, a, self.output_signal, zi=self.lowpass_states)
            self.output_signal = o
            self.lowpass_states = states

        if self.shm.apply_bandpass:
            b, a = self.create_bandpass_filter()
            # [o, states] = sig.lfilter(b, a, self.output_signal, zi=self.bandpass_states)
            [o, states] = sig.lfilter(b, a, self.output_signal, zi=self.bandpass_states)
            self.output_signal = o
            self.bandpass_states = states

        if self.shm.apply_highpass:
            b, a = self.create_highpass_filter()
            [o, states] = sig.lfilter(b, a, self.output_signal, zi=self.highpass_states)
            self.output_signal = o
            self.highpass_states = states


    def build_AM_params(self):
        # Low pass filter for AM
        self.b_am_lpf, self.a_am_lpf = sig.ellip(self.filter_order, 0.2, 50, 0.48)

        self.I = 1j  # Complex number
        s = np.array([self.I**x for x in range(0,self.filter_order+1)])

        # Complex coefficients
        self.b_am_lpf = self.b_am_lpf * s
        self.a_am_lpf = self.a_am_lpf * s


    def apply_modulation(self):
        if self.shm.apply_am:
            self.om = 2 * math.pi * self.shm.am_mod_freq / self.shm.sample_rate

            [o, states] = sig.lfilter(self.b_am_lpf, self.a_am_lpf, self.output_signal, zi=self.am_mod_states)

            self.am_mod_states = states
            for n in range(self.shm.blocklen):
                self.theta = self.theta + self.om
                self.output_signal[n] = o[n] * (math.e**(self.I*self.theta))

            self.output_signal = self.output_signal.real

            while self.theta > math.pi:
                self.theta = self.theta - 2*math.pi

            self.output_signal = np.clip(self.output_signal, -self.maxval, self.maxval)
            

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
            if self.shm.audio_loaded:
                self.create_output_signal()
                self.apply_filters()
                self.apply_modulation()
                self.output_signal = self.output_signal * self.shm.gain
                # out = self.output_signal.astype('int16').tobytes()
                # self.stream.write(out)
                # self.clean_output()
            else:
                for clip in self.slices:
                    clip.playing = False
                print("No audio loaded! Nothing to play.")
        

        out = self.output_signal.astype('int16').tobytes()
        self.stream.write(out)

        if self.recording_output:
            self.record_audio_output()

        if audio_to_play:
            if self.shm.audio_loaded:
                self.clean_output()


    def load_audio_file(self, filename):
        self.sample = wave.open(filename, 'rb')

        self.shm.sample_num_channels = self.sample.getnchannels()
        self.shm.sample_rate = self.sample.getframerate()
        self.shm.sample_num_frames = self.sample.getnframes()
        self.shm.sample_width = self.sample.getsampwidth()

        print("num_channels: {}".format(self.sample.getnchannels()))
        print("sample_width: {}".format(self.sample.getsampwidth()))

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
            self.full_source.mono_signal = self.left_chan

        self.full_source.signal = np.asarray(self.full_source.signal)
        self.full_source.signal = self.full_source.signal / 1.5
        

        if len(self.shm.clips) >= 1:
            self.shm.clips[0] = self.full_source
        else:
            self.shm.clips.append(self.full_source)

        self.maxval = 2**((self.shm.sample_width*8)-1)-1

        self.create_playback_stream()
        self.send_event(MsgType.UPDATE_PLOT)
        self.shm.audio_loaded = True

        # print("Full source shape: {}".format(self.full_source.signal.shape))

    
    def init_record_audio_input(self):
        self.width = self.shm.sample_width
        self.channels = self.shm.sample_num_channels
        self.rate = self.shm.sample_rate
        self.blocklen = self.shm.blocklen

        p = pyaudio.PyAudio()
        PA_FORMAT = p.get_format_from_width(self.width)
        self.input_stream = p.open(
            format = PA_FORMAT,
            channels = self.channels,
            rate = self.rate,
            input = True,
            output = True,
            frames_per_buffer = self.blocklen)
        
        self.send_event(MsgType.INPUT_STREAM_START)

        self.recorded_signal = []

    
    def save_recorded_audio_input(self):
        self.full_source = AudioClip()
        self.full_source.signal = np.asarray(self.recorded_signal)

        self.shm.sample_num_frames = len(self.full_source.signal)

        if len(self.shm.clips) >= 1:
            self.shm.clips[0] = self.full_source
        else:
            self.shm.clips.append(self.full_source)

        self.maxval = 2**((self.shm.sample_width*8)-1)-1

        self.create_playback_stream()
        self.send_event(MsgType.UPDATE_PLOT)
        self.shm.audio_loaded = True

        self.input_stream.stop_stream()
        self.input_stream.close()

        self.clean_output()


    def record_audio_input(self):
        sig_bytes = self.input_stream.read(self.blocklen, exception_on_overflow=False)
        input_block = np.frombuffer(sig_bytes, dtype=np.int16)
        # self.recorded_signal.extend(input_block.tolist())

        if len(input_block) == self.blocklen:
            self.output_signal = input_block
            self.apply_filters()
            self.apply_modulation()
            self.output_signal = self.output_signal * self.shm.gain
            out = self.output_signal.astype('int16').tobytes()
            self.stream.write(out)

            self.recorded_signal.extend(self.output_signal.tolist())



    def init_record_audio_output(self):
        out_filename = "output.wav"
        self.wf_out = wave.open(out_filename, 'w')
        self.wf_out.setnchannels(self.shm.sample_num_channels)
        self.wf_out.setsampwidth(self.shm.sample_width)
        self.wf_out.setframerate(self.shm.sample_rate)


    def record_audio_output(self):
        write_out = self.output_signal.astype('int16').tobytes()
        self.wf_out.writeframes(write_out)

    
    def close_record_audio_out(self):
        self.wf_out.close()


    def create_playback_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

        self.stream = self.p.open(
            format=self.shm.pa_format,
            channels=self.shm.sample_num_channels,
            rate=self.shm.sample_rate,
            input=False,
            output=True,
            frames_per_buffer=self.shm.blocklen
        )










if __name__ == "__main__":
    matplotlib.use('TkAgg')

    # root = Tk.Tk()
    # ws = Workspace(parent_instance=root)
    # root.mainloop()


    app = WorkspaceGUI()
    app.mainloop()
