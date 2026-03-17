from PIL import Image, ImageSequence
import sys
import os

# Most useful settings for my generated gifs: 64 48 576 432

def crop_gif(input_path, output_path, crop_area):
    """
    Crops an animated GIF to a specified area.

    :param input_path: Path to the input GIF file.
    :param output_path: Path to save the cropped GIF file.
    :param crop_area: A tuple of 4 ints (left, upper, right, lower) for the crop box.
    """
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"An error occurred while opening the image: {e}")
        return

    frames = []
    # Iterate over all frames in the GIF
    for frame in ImageSequence.Iterator(img):
        # Crop the current frame
        cropped_frame = frame.crop(crop_area)
        # Append a copy to the frames list (must be a copy to avoid issues)
        frames.append(cropped_frame.copy())

    # Save the frames as a new animated GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=0, # loop=0 means loop indefinitely
            duration=img.info.get('duration', 100), # preserve original frame duration if available
            optimize=False # Optimization can cause issues with cropping/transparency
        )
        print(f"Successfully cropped GIF saved to {output_path}")
    else:
        print("No frames found in the input GIF.")


# takes in crop_gif.py filename x1 y1 x2 y2
input_gif, x1, y1, x2, y2 = sys.argv[1:]

crop_box = ( int(x1), int(y1), int(x2), int(y2))

output_gif = f"../cropped_{input_gif[3:]}"

# Call the function to crop the GIF
crop_gif(input_gif, output_gif, crop_box)
