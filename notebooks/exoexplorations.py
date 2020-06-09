import numpy as np

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def lightcurve(star_radius, planet_radius, i, imsize=(200, 200)):
    area_star_fractional = []
    field = np.zeros(imsize)
    star = create_circular_mask(imsize[0], imsize[1], radius=star_radius)
    field[star] = 1.0
    area_star_total = np.sum(star)

    for x in np.arange(imsize[0]):
        planet = create_circular_mask(imsize[0], imsize[1], center=(x, imsize[1]/2+i), radius=planet_radius)
        field[star] = 1.0
        field[planet] = 0.0
        area_star_fractional.append(np.sum(field))
    
    area_star = np.array(area_star_fractional)/area_star_total
#     plt.imshow(star)

    return np.arange(imsize[0]), area_star

