#!/usr/bin/env python3
"""Check if GPU hardware decoding is available"""

import av
import subprocess
import sys

print("="*70)
print("GPU Hardware Decoding Check")
print("="*70)

# 1. Check FFmpeg codecs
print("\n1. Checking FFmpeg CUDA support...")
try:
    result = subprocess.run(['ffmpeg', '-codecs'], 
                          capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    if 'h264_cuvid' in output:
        print("   ✓ h264_cuvid (NVIDIA CUDA H.264 decoder) available")
    else:
        print("   ✗ h264_cuvid NOT available")
    
    if 'hevc_cuvid' in output:
        print("   ✓ hevc_cuvid (NVIDIA CUDA HEVC decoder) available")
    else:
        print("   ✗ hevc_cuvid NOT available")
        
except Exception as e:
    print(f"   ✗ Error checking FFmpeg: {e}")

# 2. Check FFmpeg hwaccels
print("\n2. Checking hardware acceleration methods...")
try:
    result = subprocess.run(['ffmpeg', '-hwaccels'], 
                          capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    if 'cuda' in output:
        print("   ✓ CUDA hardware acceleration available")
    else:
        print("   ✗ CUDA hardware acceleration NOT available")
        
except Exception as e:
    print(f"   ✗ Error checking hwaccels: {e}")

# 3. Test PyAV with CUDA
print("\n3. Testing PyAV CUDA decoding...")
try:
    # Try to open a stream with CUDA
    rtsp_url = "rtsp://localhost:8554/cam5"
    
    options = {
        'rtsp_transport': 'tcp',
        'hwaccel': 'cuda',
        'hwaccel_device': '0',
    }
    
    print(f"   Attempting to open: {rtsp_url}")
    container = av.open(rtsp_url, options=options, timeout=5.0)
    stream = container.streams.video[0]
    
    print(f"   ✓ Stream opened successfully")
    print(f"   Codec: {stream.codec_context.name}")
    print(f"   Size: {stream.width}x{stream.height}")
    
    # Try to decode one frame
    frame = next(container.decode(stream))
    print(f"   ✓ Frame decoded successfully")
    print(f"   Frame type: {type(frame)}")
    
    container.close()
    print("   ✓ PyAV CUDA decoding appears to be working")
    
except Exception as e:
    print(f"   ✗ PyAV CUDA test failed: {e}")
    print("\n   Trying CPU fallback...")
    try:
        options = {'rtsp_transport': 'tcp'}
        container = av.open(rtsp_url, options=options, timeout=5.0)
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        print(f"   ✓ CPU decoding works (fallback successful)")
        container.close()
    except Exception as e2:
        print(f"   ✗ CPU fallback also failed: {e2}")

# 4. Check nvidia-smi
print("\n4. Checking NVIDIA GPU status...")
try:
    result = subprocess.run(['nvidia-smi', 'dmon', '-c', '1'], 
                          capture_output=True, text=True, timeout=3)
    print("   ✓ nvidia-smi working")
    
    # Check for decoder usage
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,decoder.stats.sessionCount,decoder.stats.averageFps', 
                           '--format=csv,noheader'], 
                          capture_output=True, text=True)
    print(f"   GPU Stats: {result.stdout.strip()}")
    
except Exception as e:
    print(f"   ✗ nvidia-smi error: {e}")

print("\n" + "="*70)
print("Recommendation:")
print("="*70)
print("""
If CUDA decoding is NOT available:
  1. Check FFmpeg build: ffmpeg -hwaccels
  2. Rebuild FFmpeg with CUDA support
  3. Or use CPU decoding (remove hwaccel options)

If you want to force CPU decoding, remove these lines:
    'hwaccel': 'cuda',
    'hwaccel_device': '0',
    'hwaccel_output_format': 'cuda'
""")
print("="*70)