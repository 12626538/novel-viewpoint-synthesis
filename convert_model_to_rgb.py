import plyfile
import numpy as np
import sys,os

# Parse args
if len(sys.argv) <= 2:
    print("Usage: python3 convert_model_to_rgb.py IN_FILE OUT_FILE")
    exit()

in_file = os.path.abspath(sys.argv[1])
out_file = os.path.abspath(sys.argv[2])

if not os.path.isfile(in_file):
    print("Input file not found:", in_file)
    exit()

# Prevent overwriting output
if os.path.isfile(out_file):
    print("Output file already exists:", out_file)
    if not input("Continue? [y/N]: ").lower().strip().startswith("y"):
        exit()

# Read data
print(f"Reading model from {in_file}")
plydata = plyfile.PlyData.read(in_file)
assert 'vertex' in plydata,"'vertex' plyElement not found"
vertex = plydata['vertex']

# Get in and out properties
properties_in = {prop.name for prop in vertex.properties}
properties_out = ['x','y','z','red','green','blue']

properties_out += ['rot_1','rot_2','rot_3','rot_4','scale_1','scale_2','scale_3','opacity']
print("Output properties:", properties_out)

# For each output property, indicate what input property it comes from
mapper={'red':'color_1_1','green':'color_1_2','blue':'color_1_3'}

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Extract data
print("Converting model...")
features = []
fields = []
for prop in properties_out:
    # Get feature vector of property
    feature = vertex[mapper.get(prop,prop)]

    # Convert color
    if prop in {'red','green','blue'}:
        feature = (feature * 0.28209479177387814) + 0.5
        feature = np.clip(feature, 0, 1)
        feature = (feature * 255).astype(np.uint8)

        features.append( feature )
        fields.append( (prop, 'u1') )

    # Convert position
    else:
        features.append( feature )
        fields.append((prop,'f4'))

# properties to numpy array
features = np.stack(features, axis=1)


# Put features as a list of tuples in output
data = np.empty(features.shape[0], dtype=fields)
data[:] = list(map(tuple, features))

el = plyfile.PlyElement.describe(data, 'vertex')
plyfile.PlyData([el], text=True).write(out_file)
print(f"Model saved to {out_file}")
