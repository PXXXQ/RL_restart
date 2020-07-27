from io import BytesIO
import pickle


def serialize(data):
    f = BytesIO()
    pickle.dump(data, f)
    serialized_data = f.getvalue()
    f.close()
    return serialized_data


def deserialize(serialized_data):
    f = BytesIO()
    f.write(serialized_data)
    f.seek(0)
    data = pickle.load(f)
    f.close()
    return data

