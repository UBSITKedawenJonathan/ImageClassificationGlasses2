import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support() #i was gettning freeze error so i searched this
    from duckduckgo_search import ddg_images #ddg search engine
    from fastai.vision.all import * #provides image processing,classification, object detection
    from fastcore.all import * #support library for fastai
    from time import sleep #give your pc a break and add this
    from fastdownload import download_url #allows downloading of image and other urls

    searches = 'Pajama', 'Tuxedo'
    path = Path('Tux_or_Paj')

    """
    def search_images(term, max_images=50):
        print(f"Searching for '{term}'")
        return L(ddg_images(term, max_results=max_images)).itemgot('image')

    #Run these 2 following lines to download the searched images' url and downloads the first one via. [0], this is for testing the data later
    download_url(search_images('person in tuxedo', max_images=1)[0], 'Tuxedo.jpg', show_progress=False)
    download_url(search_images('Person in pajama', max_images=1)[0], 'Pajama.jpg', show_progress=False)
    """#This method will search images using duckduckgo

    """
    for o in searches:
        dest = (path / o) #create directory named <o>, first iteration is "Pajama" then "Tuxedo"
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo'))
        resize_images(path / o, max_size=400, dest=path / o)

    #Check if the downloaded images are good, if not delete them
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    len(failed)
    """# Execute this block atleast once then you can comment it out to prevent over downloading

    """
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),#input of imageblock, output of categoryblock
        get_items=get_image_files,#get data from path in dataloaders(path)
        splitter=RandomSplitter(valid_pct=0.2, seed=42),#20% of data will be used for test data
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)#Batch size 32
    dls.show_batch(max_n=6)#supposed to show 6 images with labels but its not showing
    learn = cnn_learner(dls, resnet18, metrics=error_rate)#gets the data from dls, uses resnet18 as the learning model, and an ouput of error_rate
    learn.fine_tune(3)#epoch amount

    is_tux, _, probs = learn.predict(PILImage.create('tux2.jpg'))#uses the learn model to predict data of the variables and store in is_tux, _ to ignore useless values?, probs is numerical value of prediction
    print(f"This is a: {is_tux}.")
    if is_tux=="Tuxedo":
        print(f"Probability it's a Tuxedo: {(1-probs[0])*100:.2f}%") #present the probability rate in percentage 2 decimals
    else:
        print(f"Probability it's a Pajama: {probs[0]*100:.2f}%") #present the probability rate in percentage 2 decimals
    """
