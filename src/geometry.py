"""All geometry is in mm. IDs are metadata, never coordinate columns."""
import hashlib
import numpy as np


def read_centers(path):
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] < 3 or not np.isfinite(raw).all():
        raise ValueError(f'{path}: expected finite ID, x, y columns')
    ids = raw[:, 0].astype(np.int64)
    if not np.array_equal(ids, raw[:, 0]) or len(np.unique(ids)) != len(ids):
        raise ValueError('PMT IDs must be unique integers')
    xy = raw[:, 1:3]
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('Duplicate PMT centers')
    return ids, xy


class CircleIndex:
    """Spatial buckets: avoid constructing photons x all-PMT distance matrices."""
    def __init__(self, centers, radius):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.radius = float(radius)
        if radius <= 0:
            raise ValueError('Radius must be positive')
        delta = self.centers[:, None] - self.centers[None, :]
        d2 = (delta ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        if np.any(d2 < (2 * radius - 1e-7) ** 2):
            raise ValueError('Photocathodes overlap; check units/radii/coordinates')
        self.buckets = {}
        for i, (x, y) in enumerate(self.centers):
            lo = np.floor((np.array([x, y]) - radius) / radius).astype(int)
            hi = np.floor((np.array([x, y]) + radius) / radius).astype(int)
            for ix in range(lo[0], hi[0] + 1):
                for iy in range(lo[1], hi[1] + 1):
                    self.buckets.setdefault((ix, iy), []).append(i)

    def locate(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite photon coordinates')
        result = np.full(len(points), -1, dtype=np.int64)
        if not len(points):
            return result
        cells, inverse = np.unique(np.floor(points / self.radius).astype(np.int64),
                                   axis=0, return_inverse=True)
        order = np.argsort(inverse)
        offsets = np.r_[0, np.cumsum(np.bincount(inverse))]
        for k, cell in enumerate(cells):
            candidates = self.buckets.get(tuple(cell), [])
            if not candidates:
                continue
            take = order[offsets[k]:offsets[k + 1]]
            d2 = ((points[take, None] - self.centers[candidates]) ** 2).sum(-1)
            closest = d2.argmin(axis=1)
            valid = d2[np.arange(len(take)), closest] <= self.radius ** 2 + 1e-9
            result[take[valid]] = np.asarray(candidates)[closest[valid]]
        return result

    def counts(self, points):
        assignments = self.locate(points)
        return np.bincount(assignments[assignments >= 0], minlength=len(self.centers))


def quadrature(lr_centers, hr_centers, lr_radius, hr_radius, samples):
    """Deterministic quadrature over the actual HR/LR intersection lenses.

    Weights integrate density relative to the mean over the LR photocathodes.
    The unobserved gaps are excluded, not supervised as zero incident flux.
    """
    if samples < 16:
        raise ValueError('Use at least 16 integration samples per HR PMT')
    CircleIndex(lr_centers, lr_radius)  # intersections cannot overlap each other
    points = np.repeat(np.asarray(hr_centers)[:, None, :], samples, axis=1).copy()
    weights = np.zeros((len(hr_centers), samples), dtype=np.float64)
    total_lr_area = len(lr_centers) * np.pi * lr_radius ** 2
    for j, center in enumerate(hr_centers):
        distances = np.linalg.norm(lr_centers - center, axis=1)
        neighbors = np.flatnonzero(distances < lr_radius + hr_radius)
        if not len(neighbors):
            continue
        if samples < len(neighbors) * 4:
            raise ValueError('Too few quadrature samples for intersecting photocathodes')
        cursor = 0
        for part, neighbor in enumerate(neighbors):
            n = samples // len(neighbors) + int(part < samples % len(neighbors))
            d = distances[neighbor]
            axis = (lr_centers[neighbor] - center) / d if d > 1e-12 else np.array([1., 0.])
            perpendicular = np.array([-axis[1], axis[0]])
            lo, hi = max(-hr_radius, d - lr_radius), min(hr_radius, d + lr_radius)
            k = np.arange(n, dtype=np.float64)
            x = lo + (k + .5) / n * (hi - lo)
            height = np.sqrt(np.maximum(0., np.minimum(hr_radius ** 2 - x ** 2,
                                                       lr_radius ** 2 - (x - d) ** 2)))
            # Midpoint integration in x, low-discrepancy samples along each y slice.
            y = (2 * np.mod((k + .5) * .6180339887498949, 1.) - 1) * height
            points[j, cursor:cursor+n] = center + x[:, None] * axis + y[:, None] * perpendicular
            local_weights = 2 * height * (hi - lo) / n
            if d <= abs(lr_radius - hr_radius):
                area = np.pi * min(lr_radius, hr_radius) ** 2
            else:
                r, big = hr_radius, lr_radius
                a = np.arccos(np.clip((d*d + r*r - big*big) / (2*d*r), -1, 1))
                b = np.arccos(np.clip((d*d + big*big - r*r) / (2*d*big), -1, 1))
                radical = max(0., (-d+r+big)*(d+r-big)*(d-r+big)*(d+r+big))
                area = max(0., r*r*a + big*big*b - .5*np.sqrt(radical))
            # Exact lens area makes constant-field integration exact, including tiny overlaps.
            weights[j, cursor:cursor+n] = local_weights * (area / local_weights.sum()) / total_lr_area
            cursor += n
    coverage = weights.sum(1) * total_lr_area / (np.pi * hr_radius ** 2)
    return points.astype(np.float32), weights.astype(np.float32), np.minimum(coverage, 1.).astype(np.float32)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
