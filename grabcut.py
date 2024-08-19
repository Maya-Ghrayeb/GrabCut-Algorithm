import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
import igraph as ig
from scipy.stats import multivariate_normal
import time

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel


graph = ig.Graph(directed=False)
ind1=True
ind2=False
beta=0
prev_energy=0
height=0
width=0
n_components=5

class gmm:
    def __init__(self,  n_components=5):
        self.means= np.zeros( n_components)
        self.weights= np.zeros(n_components)
        self.determinant= np.zeros(n_components)
        self.covariances=  [0,0,0,0,0]



#     Defining helpful functions:

def beta_calculate(img):
    global beta 
    x1 = img[:-1,1:] - img[1:,:-1]
    x2 = img[:-1,:-1] - img[1:,1:]
    x3 = img[:-1,:] - img[1:,:]
    x4 = img[:,:-1] - img[:,1:]

    numerator = np.sum(x1**2) + np.sum(x2**2) + np.sum(x3**2) + np.sum(x4**2)
    denominator = img.size - img.shape[0] - img.shape[1] + 2
    beta = 0.5 * numerator / denominator
    
    return beta

def N_links(y1, x1, y2, x2):
    
    dist = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
    diff= img[y1, x1] - img[y2, x2]
    val= 50 * np.exp(-beta * diff) / dist
    
    return val


def likelihood_calc(img, fg_gmm, bg_gmm):
    
    global height, width, n_components
    bg_probs = np.zeros((height, width, n_components))
    fg_probs = np.zeros((height,width, n_components))
    
    for i in range(n_components):
        fg_probs[:, :, i] = fg_gmm.weights[i] * multivariate_normal.pdf(img, mean=fg_gmm.means[i], cov=fg_gmm.covariances[i], allow_singular=True)
        bg_probs[:, :, i] = bg_gmm.weights[i] * multivariate_normal.pdf(img, mean=bg_gmm.means[i], cov=bg_gmm.covariances[i], allow_singular=True)

    D_foreground = -np.log(np.clip(np.sum(fg_probs, axis=2), 1e-15, None))
    D_background = -np.log(np.clip(np.sum(bg_probs, axis=2), 1e-15, None))

    return D_foreground, D_background


def update_edge(graph,v1,v2,weight):
    
    index = graph.get_eid(v1, v2)  
    edge = graph.es[index] 
    edge["weight"] = weight 


def ind(index):
        global  width
        return ((index // width), (index % width))


def id(i,j):
    
        global  width
        return width * i + j


def calc_all_Nlinks(i,j, edges, weights, img ):
    v = i * img.shape[1] + j
    if i > 0  :
        edges.append((v, (img.shape[1] * (i-1)) + j ))
        weights.append(N_links(i, j, i - 1, j))
    if j > 0 :
        edges.append((v, (img.shape[1] * i) + j-1))
        weights.append(N_links(i, j,  i, j - 1))
    if i > 0 and j > 0 :
        edges.append((v, (img.shape[1] * (i-1)) + j-1))
        weights.append(N_links(i, j, i - 1, j - 1))
    if i > 0 and j < img.shape[1] - 1  :
        edges.append((v, (img.shape[1] * (i-1)) + j+1))
        weights.append(N_links(i, j, i - 1, j + 1))
    return edges, weights
        

def calc_Tlinks(i, j, source, sink, edges, weights, img, mask, fg_D, bg_D, k):

    v = i * img.shape[1] + j

    if mask[i, j] == GC_FGD:
        edges.append((v, sink))
        weights.append(k)
        edges.append((v,source))
        weights.append(0)
               
    elif mask[i, j] == GC_BGD:
        edges.append((v,source))
        weights.append(k)
        edges.append((v,sink))
        weights.append(0)


    else:
        edges.append((v,source))
        weights.append(bg_D[i,j])
        edges.append((v, sink))
        weights.append(fg_D[i,j])
        
    return edges, weights



# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    start_time = time.time()

    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break
    elapsed_time = time.time() - start_time
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    global   height, width , beta, n_components
    
    height=img.shape[0]
    width=img.shape[1]
    beta=beta_calculate(img)

    
    fg_gmm = gmm(n_components)
    bg_gmm = gmm(n_components)
    
    return bg_gmm , fg_gmm




# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    global   height, width, n_components
    
    background_pixels = img[(mask==0) ]
    foreground_pixels = img[(mask==3) | (mask==1)]
    
   
    #k-means clustering
    background_kmeans = KMeans(n_clusters=5, random_state=0,n_init=10).fit(background_pixels)
    foreground_kmeans = KMeans(n_clusters=5, random_state=0,n_init=10).fit(foreground_pixels)

    bgGMM.means= background_kmeans.cluster_centers_
    fgGMM.means=foreground_kmeans.cluster_centers_

    for i in range(n_components):
        bg_component_pixels = background_pixels[background_kmeans.predict(background_pixels) == i]
        fg_component_pixels = foreground_pixels[foreground_kmeans.predict(foreground_pixels) == i]

        
        fgGMM.weights[i]=np.sum(foreground_kmeans.predict(foreground_pixels) == i) / (len(foreground_pixels))
        fgGMM.covariances[i]=np.cov(fg_component_pixels.T)
        bgGMM.weights[i]=np.sum(background_kmeans.predict(background_pixels) == i) / (len(background_pixels))
        bgGMM.covariances[i]=np.cov(bg_component_pixels.T)

    return bgGMM, fgGMM



def calculate_mincut(img, mask, bgGMM, fgGMM):
    global ind1 , ind2 , height, width

    min_cut = [[], []]
    
    fg_D, bg_D=likelihood_calc(img, fgGMM, bgGMM)
    k=max(np.max(fg_D),np.max(bg_D))
    source = height * width
    sink = height * width + 1

    
    # assigning edges and their weights:
    
    if(ind1):
        graph.add_vertices(height * width + 2)
        edges = []
        e_weights=[]
        for i in range(height):
            print(i)
            for j in range(width):

                #N-links
                edges, e_weights= calc_all_Nlinks(i,j, edges, e_weights, img)

                
                #T-links
                edges, e_weights= calc_Tlinks(i, j, source, sink, edges, e_weights, img, mask, fg_D, bg_D, k)

        ind1=False
        ind2=True
        graph.add_edges(edges, attributes={'weight': e_weights})

    
    #T-links update starting from second iteration:
    if(ind2):
        for i in range(height):
            print(i)
            for j in range(width):
                if (mask[i][j] == 3 or mask[i][j] == 1) :
                    update_edge(graph,i * width + j,sink,fg_D[i,j])
                    update_edge(graph,i * width + j,source,bg_D[i,j])
                else:
                   
                    update_edge(graph,i * width + j,sink,0)
                    update_edge(graph,i * width + j,source,k)     

                   
    cut = graph.st_mincut(sink, source, capacity='weight')
    min_cut[0] = cut.partition[1]
    min_cut[1] = cut.partition[0]
    
    return min_cut, cut.value



def update_mask(mincut_sets, mask):
    
    global height, width 
    source = height * width
    sink = height * width + 1
    
    for x in mincut_sets[1]:
        if x not in (source, sink):
            mask[ind(x)] = 0
            
    return mask


def check_convergence(energy):
    
    global prev_energy
    diff=prev_energy -energy
    
    if( (diff >=0)  and (diff < 2500)):
        convergence = True
    else:
        convergence=False
        
    prev_energy=energy
    return convergence


def cal_metric(predicted_mask, gt_mask):
    #Accuracy
    correct_pixels = np.sum(predicted_mask == gt_mask)
    all_pixels = predicted_mask.size
    accuracy = correct_pixels / all_pixels

    #Jaccard similarity
    intersection = np.sum(np.logical_and(predicted_mask, gt_mask))
    union = np.sum(np.logical_or(mask, gt_mask))
    jaccard = intersection / union

    return accuracy, jaccard

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='teddy', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
