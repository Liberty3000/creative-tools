def video(frames, output_file, fps=14):
    import imageio, tqdm

    writer = imageio.get_writer(output_file, fps=fps)
    for im in tqdm.tqdm(frames):
        try: writer.append_data(imageio.imread(im))
        except: pass
    writer.close()

