from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('../yolov8n.pt')

    results = model.train(
        data='logodet3k_yolo/data.yaml',
        epochs=40,
        imgsz=640,
        batch=32,
        name='logodet3k_yolov8s_baseline50',
        patience=0,
        save=True,
        device=0,
        workers=2,   
    )


    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}")
