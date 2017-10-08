import bvh_reader as br
import bvh_writer as bw

motion, time_step, batch_Frame, frame_Time ,file_directory=br.Motion_Data_Preprocessing(time_step=90, batch_Frame=1)
bw.Motion_Data_Making(motion, time_step, batch_Frame, frame_Time , file_directory)