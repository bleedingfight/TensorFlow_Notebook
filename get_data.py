import requests
def get_data(url):
    request = requests.get(url)
    name = url.split('/')[-1]
    if request.status_code== 200:
        with open(name,'w') as f:
            f.write(request.content.decode('utf-8'))
    else:
        print('sorry,network unstable')
def main():
    url = 'https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/10165151/macbeth.txt'
    get_data(url)
if __name__ == '__main__':
    main()
