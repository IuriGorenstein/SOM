import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import RegularPolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs                   # import projections
import cartopy.feature as cf                 # import features
import copy
from google.cloud import storage
import sys
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import time
from ipywidgets import interact
impprt math

def mask_2D(vector,lim):
    #uses 2D vector to create a lat/lon mask where the vector has values under 'lim'
    
    return ( vector < lim )

def Compress_data_2D(vector,mask):
   # vetoriza matriz de dim=dimensao para 2 dimensoes de acordo com mascara espacial
    # a ideia é, se existem 3 dimenoes, 1é temporal e 2 espaciais
    # se existem 4, uma é o modelo, uma temporal e 2 espaciais.
    vector=np.array(vector)
    
    vetor=vector[:,mask]
    
    return vetor

def Plot_SST_map(SST,lat,lon):
    fig = plt.figure(figsize=(18,3.5))
    ax= fig.add_subplot(1, 1, 1, aspect='equal',
                         projection=ccrs.PlateCarree(central_longitude=-90.0, globe=None))
    
    #ax.set_extent([-65, 25, -49, 29.5], ccrs.PlateCarree()) # Atlantic Ocean
    
        # SST #
        
    levels = np.linspace(-1.5,1.5, 90)
    im = ax.contourf(lon, lat, SST,levels=levels, cmap="RdBu_r",extend='both',transform=ccrs.PlateCarree()) ## MUDAR PARA AZUL E VERMELHO 
    
        # SST #       
    ax.coastlines(linewidths=1)
    ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
    
    return

def MOD_anomaly(vector,lat,lon):
    # calcula a anomalia dos diferentes modelos dentro do vetor
    
    anomaly=copy.deepcopy(vector)

    # Climatology 
    end=len(vector)
    mean=copy.deepcopy(vector[0:12,:,:])
    for j in range(12):
        mean[j,:,:]=np.nanmean(vector[j:end:12,:,:], axis=0)
    
    #CALCULO DO VETOR ANOMALIA
    for t in range(12):
        anomaly[t:end:12,:,:]=vector[t:end:12,:,:]-mean[t,:,:]
    
    return anomaly

def smooth(x,window_len=1,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2-1):-(window_len//2)]

def BMU(porcentagem):
    # Finds BMU from porcentage matrix
    
    tempo=len(porcentagem[0])
    BMU=np.zeros((2,tempo))
    for t in range(tempo): # tempo da série temporal
        
        # # # # # # # # # Se porcentagem é um vetor estritamente positivo tenho que filtrar os passos temporais que não tem correlacao com nenhum padrao. Caso contrario o padrao 0 será sempre escolhido
        #pula=True
        #for padrao in range(porcentagem.shape[0]):
        #    if porcentagem[padrao,t] > 0:
        #        pula = False # Se há algum valor for não nulo, busco o BMU desse tempo
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #if pula:
        #    BMU[0,t]=None
        #    BMU[1,t]=None
        #else:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
        BMU[0,t]=porcentagem[:,t].argmax()
        BMU[1,t]=np.max(porcentagem[:,t])
        
    return BMU


def Correlacao(som,data):
    # Correlation from data with som neurons using the r2_score function from sklearn.
    # Returns the percentage correlation from each neuron at each time from the data.
    # OBS: the data should be 3 dimensional: time/lat/lon
    
    codebook2 =  som.codebook.matrix
    me, st= np.mean(data, axis=0), np.std(data, axis=0)
    codebooksst =  codebook2*st[:]+me[:]

    n_clusters=Find_Clusters(som,2,20, verbose=1, mode='silhouette') # 'silhouette', 'elbow' ou 'both'
      
    tempo_inicio=0
    tempo_final=data.shape[0]
    tempo=(tempo_final-tempo_inicio)

    # correlações positivas
    porcentagem = np.zeros((len(codebook2),data.shape[0]))# porcentagem de variancia explicada por cada padrão no vetor maior.
    #a=0  
    k=0
    start = time.time()
    for i in progressbar(range(tempo), "Correlaçao: ", 40):
        for j in range(len(codebooksst)):
            correlacao=r2_score(codebooksst[j], data[i])
            # # # # # # # # #  PEGA APENAS VALORES > 0 DE PORCENTAGEM #
            if correlacao < 0 :                                      #  
                a=1                                                  #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            else:
                porcentagem[j,i]=correlacao
                
        k=k+1
    end = time.time()
    minu = ("{:.0f}".format((end - start)//60))
    sec = ("{:.0f}".format((end - start)%60))
            
    print('Time elapsed: ' + str(minu) + '\'' + str(sec) + '\"')
        
        
    ################ ENCONTRA ERRO ###############################
    #    if i == 100:
    #        if a == 0:
    #            print('ERRO: Nenhuma correlacao encontrada nos 100 primeiros dados.')
    #            return
 
    #if a == 0:
    #    print('ERRO: Nenhuma correlacao encontrada na variável: ' + variavel +'.')
    #    return
    ##############################################################
    
            
            
    return codebooksst,porcentagem, n_clusters

def Find_Clusters(som,Clustermin,Clustermax, verbose=1, mode='silhouette'): # Clustermin >=2
    # Finds number of CLUSTERS
    if verbose==1:
        print("Encontrando número de clusters")
    codebook2 =  som.codebook.matrix
    
    # Vetor do Erro Quadrático médio
    sse = []
    # Vetor Silhouette
    silhouette_coefficients = []
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42,}

    for k in progressbar(range(Clustermin,Clustermax), "Buscando Clusters: ", 40):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(codebook2)
        score = silhouette_score(codebook2, kmeans.labels_)
        silhouette_coefficients.append(score)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(Clustermin, Clustermax), sse, curve="convex", direction="decreasing")
    
    # Cotovelo
    elbow=kl.elbow
    # Silhouette
    silhouette=np.argmax(silhouette_coefficients)
    
    if mode == 'silhouette':
        return silhouette
    elif mode == 'both':
        return elbow, silhouette
    elif mode == 'elbow':
        return elbow
    
def plot_hex_maps_SST(d_matrix,clusters,porcentagem,ano,PATTERNsst,latsst,lonsst, titles=[], shape=[1, 2], comp_width=5, hex_shrink=1.0, fig=None,
                 colorbar=True):
    """
    Plot hexagon map where each neuron is represented by a hexagon. The hexagon
    color is given by the distance between the neurons (D-Matrix)
    Args:
    - grid: Grid dictionary (keys: centers, x, y ),
    - d_matrix: array contaning the distances between each neuron
    - w: width of the map in inches
    - title: map title
    Returns the Matplotlib SubAxis instance
    """
    
    ###################### AJUSTA LAT E LON PARA ATLANTICO ##############################

    #CORTES ATLANTICO
    ## de 65W à 25E - 50S à 30N
    #lata=latsst[40:120]
    #lona=np.ma.concatenate((lonsst[0:25],lonsst[295:360]),axis=0)       
    
    #####################################################################################

    d_matrix = np.flip(d_matrix, axis=0)

    def create_grid_coordinates(x, y):
        coordinates = [x for row in -1 * np.array(list(range(x))) for x in
                       list(zip(np.arange(((row) % 2) * 0.5, y + ((row) % 2) * 0.5), [0.8660254 * (row)] * y))]
        return (np.array(list(reversed(coordinates))), x, y)

    if d_matrix.ndim < 3:
        d_matrix = np.expand_dims(d_matrix, 2)

    if len(titles) != d_matrix.shape[2]:
        titles = [""] * d_matrix.shape[2]

    n_centers, x, y = create_grid_coordinates(*d_matrix.shape[:2])

    # Size of figure in inches
    if fig is None:
        xinch, yinch = comp_width * shape[1], (comp_width * (x / y) * shape[0]) /2
        fig = plt.figure(figsize=(xinch, yinch), dpi=72.)

    for comp, title in zip(range(d_matrix.shape[2]), titles):
        ax = fig.add_subplot(1, 3, comp + 1, aspect='equal')

        # Get pixel size between two data points
        xpoints = n_centers[:, 0]
        ypoints = n_centers[:, 1]
        ax.scatter(xpoints, ypoints, s=0.0, marker='s')
        ax.axis([min(xpoints)//1 - 1., np.max(xpoints)//1 + 1.,
                 min(ypoints)//1 - 1., np.max(ypoints)//1 + 1.])
        xy_pixels = ax.transData.transform(np.vstack([xpoints, ypoints]).T)
        xpix, ypix = xy_pixels.T

        # discover radius and hexagon
        apothem = hex_shrink * (xpix[1] - xpix[0]) / math.sqrt(3)
        area_inner_circle = math.pi * (apothem ** 2)
        dm = d_matrix[:, :, comp].reshape(np.multiply(*d_matrix.shape[:2]))
        
        # NEURON MAP WITH CPOSITIVE CORRELATIONS IN BLACK ########################################
        
        for i in range(len(clusters)):
            
            collection_bg = RegularPolyCollection(
                numsides=6,  # a hexagon
                rotation=0,
                sizes=(area_inner_circle,),
                array=dm,
                cmap="binary",
                offsets=n_centers[i],
                transOffset=ax.transData,
                alpha=(0.1+porcentagem[i])/1.1 # SOMBRA
            )
            ax.add_collection(collection_bg, autolim=True)
        
        ax.axis('off')
        ax.autoscale_view()
        ax.set_title(title)#, fontdict={"fontsize": 3 * comp_width})
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(collection_bg, cax=cax)
        if not colorbar:
            cbar.remove()
        
        comp=comp+1 #next plot
    
        # SST MAP POT
        
        # SST filter according to percentage matrix #########################################
              
        #cbar.ax.tick_params(labelsize=3 * comp_width)
        masc = (porcentagem > 2) # cria uma mascara de 800 False.
        a=0 # marca se há alguma correlação positiva ou não na matriz.
        for i in range(porcentagem.shape[0]):
            if porcentagem[i] > 0:
                masc[i] = True    # para os valores de correlação positiva (porcentagem[i] > 0) fica True
                a=1
                
        if a==0:
            return
        
        # Filtrando Padrões que tem correlção
        filtradosst = PATTERNsst[masc]        
        for i in range(filtradosst.shape[1]):
            for j in range(filtradosst.shape[2]):
                filtradosst[:,i,j]=filtradosst[:,i,j]*porcentagem[masc]
              
        ay = fig.add_subplot(1, 6, comp + 2, aspect='equal',
                             projection=ccrs.PlateCarree(central_longitude=-90.0, globe=None))
        plt.title("year " + str(ano) ,size=15) 
        ay.set_extent([-65, 25, -49, 29.5], ccrs.PlateCarree())
            # SST #
        levels = np.linspace(-0.5,0.5, 90)
        im = ay.contourf(lonsst, latsst, np.nanmean(filtradosst,0),levels=levels,cmap="RdBu_r",extend='both',transform=ccrs.PlateCarree()) ## MUDAR PARA AZUL E VERMELHO TRADICIONAL cmap='bwr'
            # SST #
               
        ay.coastlines(linewidths=1)
        ay.add_feature(cf.BORDERS, linestyle=':', alpha=.5) 
              
        
    return ax, list(reversed(n_centers))

def Imprime_evolucao_SST(som,clusters,porcentagem,PATTERNsst,latsst,lonsst,t):
    
    plt.rcParams.update({'figure.max_open_warning': 0}) # evita warning que aparece com mais de 20 plots
    
    tempo=porcentagem.shape[1]
    xx=np.zeros(tempo)
    color=['a']*mapsize[0]*mapsize[1]
    for i in range(tempo):
        xx[i]=i/12
        
    from sompy.visualization.plot_tools import plot_hex_map
    msz = som.codebook.mapsize
    cmap="jet"
    #for t in progressbar(range(tempo), "Printing: ", 40):
    fig = plt.figure(figsize=(40, 120), dpi=72.)
    ano=xx[int(t)]//1

    a= plot_hex_maps_SST(np.flip(clusters.reshape(msz[0], msz[1])[::], axis=0),clusters,porcentagem[:,t],ano,PATTERNsst,latsst,lonsst, colorbar=False,fig=fig)
    
def EvolPlot(t):
    Imprime_evolucao_SST(som,cluster,Prct,PATTERN,Alat,Alon,t)
    
def progressbar(it, prefix="", size=40, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        space=int((size-18)/2)
        a='#'*space
        mensagem = a + ' VAI CORINTHIANS! ' + a
        file.write("%s[%s%s] %i/%i\r" % (prefix, mensagem[0:x], "."*(size-x), j, count)) #"#"*x
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()