import torch


'''#numpy version
def get_image_pointset(rendering_images, npoint=cfg.CONST.NPOINT):
    batch_size = imgs.shape[0]
    # search_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    all_points = np.empty(shape=(batch_size, npoint, 2), dtype=int)
    for index, img in enumerate(imgs):
        img_points = img >= search_value
        img_points = np.where(img_points[..., 0] & img_points[..., 1] & img_points[..., 2])
        img_points = np.concatenate([np.expand_dims(coord, axis=1) for coord in img_points], axis=1)
        sample_points = farthest_point_sample(torch.from_numpy(img_points), npoint)
        all_points[index] = np.expand_dims(img_points[sample_points], axis=0)
        sample_image = np.zeros(shape=(height, width), dtype=np.uint8)
        sample_image[image_points[sample_points][:, 0], image_points[sample_points][:, 1]] = 255
        cv2.imshow("img", sample_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return all_points'''


def get_image_pointset(image, npoint=2048):
    """
    Input:
        image: images data, [height, width, channel]
        npoint: number of samples
    Return:
        all_points: sampled points, [npoint, 2]
    """
    # height, width, channel = image.shape
    # search_value = torch.tensor(data=[1.0, 1.0, 1.0], dtype=torch.float32)
    # all_points = torch.zeros(size=(batch_size, npoint, 2), dtype=torch.float32)
    image_points = torch.where(torch.mean(input=image, dim=2) >= 0.5)
    image_points = torch.stack(image_points, dim=1)
    sample_points = farthest_point_sample(image_points, npoint)
    return image_points[sample_points]


def farthest_point_sample(xy, npoint=2048):
    """
    Input:
        xy: pointcloud data, [N, 2]
        npoint: number of samples
    Return:
        centroids: sampled point index, [npoint]
    """
    N, C = xy.shape
    centroids = torch.zeros(size=(npoint,), dtype=torch.long)  # .to(device)
    distance = (torch.ones(N) * 1e10).long()
    farthest = torch.randint(low=0, high=N, size=(1,), dtype=torch.long)
    if N > npoint:  # 2D图像的有效像素点数超过取样点数
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xy[farthest, :].view(1, 2)
            dist = torch.sum(input=(xy - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]
    else:  # 像素值不足取样
        centroids[0:N] = torch.randperm(N)  # 随机打乱原像素排序,保证取到所有有效值
        centroids[N:npoint] = torch.randint(low=0, high=N, size=(npoint - N,))  # 剩余取样点,随机取值
    return centroids
