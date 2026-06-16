#!/usr/bin/env python3
"""
Generate a GEMM (Matrix Multiply) SavedModel for TF Serving.

If the locally-built TF wheel has symbol issues, run in a clean conda env:

  conda create -n tf_gen python=3.10 -y
  conda activate tf_gen
  pip install tensorflow-cpu     # standalone, no GPU/CUDA deps
  python3 generate_model.py
  conda deactivate
"""
import os
import sys


def main(output_dir: str = 'gemm_model'):
    try:
        import tensorflow as tf
    except ImportError as e:
        sys.exit(
            f"Cannot import tensorflow: {e}\n\n"
            "Quick fix (clean conda env, no GPU/CUDA needed):\n"
            "  conda create -n tf_gen python=3.10 -y\n"
            "  conda activate tf_gen\n"
            "  pip install tensorflow-cpu\n"
            "  python3 generate_model.py\n"
        )

    class GEMMModel(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([None, None], tf.float32, name='A'),
            tf.TensorSpec([None, None], tf.float32, name='B'),
        ])
        def __call__(self, A, B):
            return {'output': tf.linalg.matmul(A, B)}

    save_path = os.path.join(output_dir, '1')
    tf.saved_model.save(GEMMModel(), save_path)
    print(f"SavedModel written to: {os.path.abspath(save_path)}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='gemm_model')
    main(p.parse_args().output)
