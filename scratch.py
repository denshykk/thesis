def process_video(file_name, output_file_name, device='mps'):
    print(f"Video processing using {device}...")
    start_time = time.time()

    model = load_model(device)
    cap = cv.VideoCapture(file_name)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file_name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(model, frame)
        out.write(processed_frame)

    cap.release()
    out.release()

    end_time = time.time()
    total_processing_time = end_time - start_time
    fps = frame_count / total_processing_time
    print("Time taken: {}".format(total_processing_time))
    print("FPS: {}".format(fps))

    return total_processing_time, fps
