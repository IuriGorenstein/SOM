import sompy
import numpy as np

def Patterns_SOM(data,som,lat,lon,mask): 
    nx=lon.shape[0] # - dimensão da imagem real, com pontos nan.
    ny=lat.shape[0] # - dimensão da imagem real, com pontos nan.
    
    ##### SEPARA CODEBOOKS ####################
    codebook2 =  som.codebook.matrix
    me, st= np.mean(data, axis=0), np.std(data, axis=0)

    codebook =  codebook2*st[:]+me[:]    
    
    PATTERN=np.zeros((mapsize[0]*mapsize[1],ny,nx))
    
    for i in range(len(PATTERN)):
        PATTERN[i,mask]=codebook[i]   
        
    return PATTERN

def Atlantic_FullSOM(data,x=30,y=20,train_len_factor=1):

    #    rough= 10
    #    finetune= 3
    #    train_rough_radiusin=None
    #    train_rough_radiusfin=None
    #    train_finetune_radiusin=None
    #    train_finetune_radiusfin=None
    
    global mapsize
    mapsize = [x,y]
    
    # SOM ################################################################################################

    som = sompy.SOMFactory.build(data, mapsize, mask=None, mapshape='planar', lattice='hexa', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods
        ## obs: initialization='pca' deveria convergir mais rapido, porem nao esta sequer convergindo
        ## obs: neighborhood='gaussian' gera resultados mais suavizados (menos realistas) porém com os mesmos padrões (Yonggang Liu et al., 2006a), a melhor deveria ser uma chamada 'epanechicov' (ep), mas ainda não implementada nessa biblioteca do SOMPY
        ## obs: map size maior aumenta o detalhe das transições entre os padrões.
        ## obs: lattice='hexa' gera padrões melhores que 'rect' (possui mais nodes e cria padrões que se assemelham melhor aos dados : menores r^2)
    print("Start Train")
    som.train(verbose='info',train_len_factor=train_len_factor) # train_rough_len=rough, train_rough_radiusin=train_rough_radiusin,train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=finetune,train_finetune_radiusin=train_finetune_radiusin,train_finetune_radiusfin=train_finetune_radiusin
    print("Finish Train")

    #sompy.mapview.View2DPacked(5, 10, 'test',text_size=8).show(som, what='codebook', which_dim=[0,1,2,3,4], cmap=None, col_sz=6) #which_dim='all' default
    
    return som

def Plot_neurons(PATTERN,Alat,Alon,x,y):
    fig = plt.figure(figsize=(y*9,x*3))
    n=1
    
    for p in progressbar(range(PATTERN.shape[0]-1,-1,-1), "Ploting neuron Patterns: ", 40):

        ## PLOT #########################################
        #

        ax = fig.add_subplot(x, y, n, aspect='equal',
                             projection=ccrs.PlateCarree(central_longitude=-90.0, globe=None))

        ax.set_extent([-65, 24, -49, 29.5], ccrs.PlateCarree())
        levels = np.linspace(-0.5,0.5, 90)
        im = ax.contourf(Alon, Alat, PATTERN[p],levels=levels,cmap="RdBu_r",extend='both',transform=ccrs.PlateCarree()) ## MUDAR PARA AZUL E VERMELHO TRADICIONAL cmap='bwr'         
        ax.coastlines(linewidths=1)
        ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
        n=n+1
    print('All done, the maps might take a while to appear')