
import onnxruntime as ort
import numpy as np

def get_onnx(model_path):
  def get_sess(model_path: str):
    providers = [
    (
      "CUDAExecutionProvider",
      {
      "device_id": 0,
      "gpu_mem_limit": int(
        8000 * 1024 * 1024
      ),  # parameter demands value in bytes
      "arena_extend_strategy": "kSameAsRequested",
      "cudnn_conv_algo_search": "HEURISTIC",
      "do_copy_in_default_stream": True,
      },
    ),
    "CPUExecutionProvider",
    ]
    sess_opts: ort.SessionOptions = ort.SessionOptions()
    sess_opts.log_severity_level = 2
    sess_opts.log_verbosity_level = 2
    sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
    return sess

  
  sess = get_sess(model_path=model_path)
  output_names = [out.name for out in sess.get_outputs()]
  input_name = sess.get_inputs()[0].name

  def run(input):
    pred = sess.run(output_names, {input_name: input})[0]
    return pred
    
  return run
    