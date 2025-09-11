# week2.py 파일의 전체 내용을 이 코드로 바꿔주세요.

try:
    import torch
    print("✓ PyTorch is successfully imported!")
    print(f"PyTorch version: {torch.__version__}")

    # 변경점 1: GPU 장치 설정
    # CUDA 대신 Apple Silicon의 MPS를 확인합니다.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ MPS is available! Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("MPS is not available. Using CPU.")

    # 변경점 2: 텐서를 지정된 장치(GPU 또는 CPU)로 이동
    # .to(device)를 추가하여 텐서를 GPU로 보냅니다.
    test_tensor = torch.tensor([1, 2, 3, 4, 5]).to(device)
    
    print(f"✓ Test tensor created: {test_tensor}")
    print(f"Tensor shape: {test_tensor.shape}")
    print(f"Tensor device: {test_tensor.device}") # 이 부분이 'mps'로 나와야 합니다.

    # Basic tensor operation
    squared_tensor = test_tensor ** 2
    print(f"✓ Basic operation test (squaring): {squared_tensor}")

    print("\n🎉 PyTorch is working correctly on the specified device!")

except ImportError as e:
    print("❌ PyTorch is not installed or not available")
    print(f"Error details: {e}")
    print("Please install PyTorch using: pip install torch")

except Exception as e:
    print(f"❌ An error occurred while testing PyTorch: {e}")    
    
    
    
