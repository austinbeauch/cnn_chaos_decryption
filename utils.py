import numpy as np

def keygen(chaos_func, length, *args):
    xn = chaos_map(chaos_func, length, *args)
    key = np.around(xn*1e9) % 256
    return key.astype(int)

def chaos_map(func, length, x0, *args, **kwargs):

    generated_map = [x0]
    
    x_n = x0

    for i in range(length):
        x_next = func(x_n, *args)
        generated_map.append(x_next)
        x_n = x_next
        
    return np.array(generated_map[:-1])


def diffuse(key, image, inv=False):
    ciphertext = np.bitwise_xor(key, image.flatten()) % 256
    return ciphertext


def encrypt(image, key, iterations=1):
    confused = arnold_cat(image, iterations)
    ciphertext = np.bitwise_xor(key, confused.flatten()) % 256
    return ciphertext.reshape(image.shape)


def arnold_cat(img, iterations=1, inv=False):
    try:
        image = img.clone()
    except AttributeError:
        image = img.copy()
        
    N = image.shape[0]
    x,y = np.meshgrid(range(N), range(N))
    xmap = (2*x+y) % N
    ymap = (x+y) % N
    
    if inv:
        xmap = (2*x-y) % N
        ymap = (-x+y) % N
        
    for i in range(iterations):
        image = image[xmap, ymap]
        
    return image

def decrypt(ciphertext, key, iterations=1):
    inv_diffusion = np.bitwise_xor(ciphertext.flatten(), key) % 256
    recovered = arnold_cat(inv_diffusion.reshape(ciphertext.shape), iterations, inv=True)
    return recovered


def improved_sine(x, alpha):
    assert 2 < alpha 
    coeff = (alpha-2) / (alpha * (1 - np.sin(np.pi/alpha) ) )
    const = (alpha-1) / alpha
    return coeff * (np.sin(np.pi*x) - 1) + const

def sine_map(x, sigma):
    return sigma * np.sin(np.pi * x)

def logistic_map(x, r):
    return r * x * (1-x)