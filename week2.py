try:
    import torch
    print("✓ PyTorch is successfully imported!")
    print(f"PyTorch version: {torch.__version__}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ MPS is available! Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("MPS is not available. Using CPU.")

    test_tensor = torch.tensor([1, 2, 3, 4, 5]).to(device)
    
    print(f"✓ Test tensor created: {test_tensor}")
    print(f"Tensor shape: {test_tensor.shape}")
    print(f"Tensor device: {test_tensor.device}")

    squared_tensor = test_tensor ** 2
    print(f"✓ Basic operation test (squaring): {squared_tensor}")

    print("\n🎉 PyTorch is working correctly on the specified device!")

except ImportError as e:
    print("❌ PyTorch is not installed or not available")
    print(f"Error details: {e}")
    print("Please install PyTorch using: pip install torch")

except Exception as e:
    print(f"❌ An error occurred while testing PyTorch: {e}")    