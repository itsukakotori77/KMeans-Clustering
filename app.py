# Machine Learning
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

# Package
from flask import Flask, json, render_template, request, make_response, redirect, Response, session
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from jinja2 import Markup
from flask import jsonify

# Class View
from flask.views import View
from flask_appbuilder import AppBuilder, expose, BaseView
from flask import session, url_for

# Import File
import os
import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go

import json
import io
import csv
import math

# Name App
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSION = set(['json', 'csv'])
app.secret_key = 'secret'

# APP Config
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'

# Path to .env file
# dotenv_path = join(dirname(__file__), '.env')  
dotenv_path = '.env'  
load_dotenv(dotenv_path)

# Konfigurasi File Upload
# Dashboard
class Dashboard:
    
    # Route Home
    @staticmethod
    @app.route('/')
    def home():

        return render_template('layouts/app.html', title='Dashboard')

class Dataset:
    @staticmethod
    @app.route('/dataset/route',  methods = ['POST'])
    def sessionUnset():
        if 'route' in session:
            return session.pop('route')

    @staticmethod
    @app.route('/dataset', methods=['GET', 'POST'])
    def datasetIndex():

        if 'route' in session:
            dataset = pd.read_csv(session.get('route'))
            return render_template('/dataset/index.html', table=Dataset.processData(dataset.values.tolist()), title='Dataset')

        if request.method == 'POST':
            # Var file
            fileData = request.files['importCSV']

            # Cek jenis request
            if 'importCSV' not in request.files:
                return render_template('/dataset/index.html', title='Dataset');

            # Cek null file
            if fileData.filename == '':
                return render_template('/dataset/index.html', title='Dataset');

            # Simpan File
            if fileData and Dataset.cekFile(fileData.filename):
                filename = secure_filename(fileData.filename)
                fileData.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # Parsing csv/json file to HTML
                dataset = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                session['route'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                return render_template('/dataset/index.html', table=Dataset.processData(dataset.values.tolist()), title='Dataset')
                
            else:
                return 'File Gagal Diupload'
        else:
            return render_template('/dataset/index.html', title='Dataset')

    @staticmethod
    def cekFile(filename):

        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

    def processData(array_data):
        data = []
        kunjungan = ''
        dekat = ''
        jarak = ''

        for row in array_data:
            # if row[6] == 1:  
            #     kunjungan = 'Supermarket'
            # elif row[6] == 2:  
            #     kunjungan = 'Mall'
            # elif row[6] == 3:
            #     kunjungan = 'Departement Store'
            # else:  
            #     kunjungan = 'Hypermarket'

            
            # if row[7] == 1:  
            #     dekat = 'Supermarket'
            # elif row[7] == 2:  
            #     dekat = 'Mall'
            # elif row[7] == 3:
            #     dekat = 'Departement Store'
            # else:  
            #     dekat = 'Hypermarket'
            # jarak = row[6] + ' KM'


            data.append([row[0], row[1], row[2], row[3], Dataset.convertRupiah(row[4]), Dataset.convertRupiah(row[5]), row[6], row[7]])

        return data

    def convertRupiah(uang):
        y = str(uang)
        if len(y) <= 3 :
            return 'Rp ' + y     
        else :
            p = y[-3:]
            q = y[:-3]
            return Dataset.convertRupiah(q) + '.' + p

class Perhitungan:
    # ====== Perhitungan ======
    @staticmethod
    @app.route('/perhitungan')
    def perhitungan():

        if 'route' in session:
            air = pd.read_csv(session.get('route'))
            dataset = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])

            # Normalisasi
            air_x = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])
            normalized = preprocessing.normalize(dataset)
            lists = normalized.tolist()

            # Normalisasi MinMax
            x_array = np.array(normalized)
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x_array)
            x_scaled.tolist()

            # Cluster
            # kmeans = KMeans(n_clusters = 4, random_state=123)
            # kmeans.fit(x_scaled)
            # centroid = kmeans.cluster_centers_

            # Elbow Method
            distortions = []
            K = range(1,10)
            for k in K:
                kmeanModel = KMeans(n_clusters=k)
                kmeanModel.fit(dataset)
                distortions.append(kmeanModel.inertia_)

            range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
            silhouette_avg = []
            silhouette_avg.insert(0, 0)
            for num_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(dataset)
                cluster_labels = kmeans.labels_
                # silhouette_avg[0] = 0
                silhouette_avg.append(silhouette_score(dataset, cluster_labels))

            return render_template('perhitungan/index.html', title='Perhitungan (Normalisasi)', rows=lists, normalize=x_array.tolist(), distortions=distortions, silhouette=silhouette_avg, range=range_n_clusters)

        return render_template('perhitungan/index.html', title='Perhitungan')

    @staticmethod
    @app.route('/perhitungan/k')
    def perhitunganCluster():

        air = pd.read_csv(session.get('route'))
        dataset = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])

        # Normalisasi
        air_x = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])
        normalized = preprocessing.normalize(dataset)
        lists = normalized.tolist()

        # Normalisasi MinMax
        x_array = np.array(normalized)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x_array)
        x_scaled.tolist()

        # Cluster
        _K = int(request.args.get('K'))
        _iteration = int(request.args.get('Jumlah_Iterasi'))

        # Clustering
        session['K'] = _K
        session['Iteration'] = _iteration

        kmeans = KMeans(n_clusters = _K, random_state=123)
        kmeans.fit(x_scaled)
        centroid = kmeans.cluster_centers_

        # Elbow Method
        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(dataset)
            distortions.append(kmeanModel.inertia_)

        range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
        silhouette_avg = []
        silhouette_avg.insert(0, 0)
        for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(dataset)
            cluster_labels = kmeans.labels_
            # silhouette_avg[0] = 0
            silhouette_avg.append(silhouette_score(dataset, cluster_labels))

        return render_template('perhitungan/index.html', title='Perhitungan (Normalisasi)', rows=lists, normalize=x_array.tolist(), centroids=centroid.tolist(), distortions=distortions, silhouette=silhouette_avg, range=range_n_clusters)

            

        # return True


class Clustering:

    # ====== Clustering ======
    @staticmethod
    @app.route('/clustering')
    def clustering():

        if 'route' in session:
            air = pd.read_csv(session.get('route'))
            dataset = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])

            # Normalisasi
            air_x = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])
            normalized = preprocessing.normalize(dataset)
            lists = normalized.tolist()

            # Normalisasi MinMax
            x_array = np.array(normalized)
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x_array)
            x_scaled.tolist()

            # Cluster
            clustering = 4

            # Cluster (Cluster 3)
            kmeans = KMeans(n_clusters = clustering, random_state=123)
            kmeans.fit(x_scaled)

            # Cluster
            air['kluster'] = kmeans.labels_
            # cluster = 0
            data_fix = 0
            cluster = Clustering.countData(air['kluster'])
            # data_fix = Clustering.countTerritory(air['Daerah'].values.tolist(), air['kluster'].values.tolist(), clustering)

            # return render_template('clustering/index.html', title='Cluster', table=Clustering.processData(air.values.tolist()), cluster=cluster, data_daerah=data_fix)
            return jsonify(cluster)

        return render_template('clustering/index.html', title='Clustering')

    @staticmethod
    @app.route('/clustering/download')
    def convertCSV():
        
        # Try Catch
        air = pd.read_csv(session.get('route'))
        dataset = air.drop(columns=['No', 'Tanggal Jatuh Tempo', 'Daerah', 'Kode_Kecamatan', 'Kode_Kelurahan'])
        result = dataset.values.tolist()
        output = io.StringIO()
        writer = csv.writer(output)
        
        line = ['Luas Bumi SPPT', 'Luas Bangunan SPPT', 'NJOP_BUMI', 'PPB yang harus dibayar', 'SPPT yang dibayar', 'Denda SPPT']
        writer.writerow(line)

        # Loop Data
        for row in result:
            line = [str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(row[3]) + ',' + str(row[4]) + ',' + str(row[5])]
            writer.writerow(line)

        # Output
        output.seek(0)
        
        # Return Value
        return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=dataset_report.csv"})
        # return jsonify(result)

    def processData(array_data):
        data = []
        kunjungan = ''
        dekat = ''
        cluster = ''

        for row in array_data:
            # if row[6] == 1:  
            #     kunjungan = 'Supermarket'
            # elif row[6] == 2:  
            #     kunjungan = 'Mall'
            # elif row[6] == 3:
            #     kunjungan = 'Departement Store'
            # else:  
            #     kunjungan = 'Hypermarket'

            
            # if row[7] == 1:  
            #     dekat = 'Supermarket'
            # elif row[7] == 2:  
            #     dekat = 'Mall'
            # elif row[7] == 3:
            #     dekat = 'Departement Store'
            # else:  
            #     dekat = 'Hypermarket'
            
            # if row[8] == 0:  
            #     cluster =  1 #'Supermarket'
            # elif row[8] == 1:  
            #     cluster = 2 #'Mall'
            # elif row[8] == 2:
            #     cluster = 3 #'Departement Store'
            # else:  
            #     cluster = 4 #'Hypermarket'

            data.append([row[0], row[1], row[2], row[3], Dataset.convertRupiah(row[4]), Dataset.convertRupiah(row[5]), row[6], row[7], row[8]])

        return data

    def countData(array):

        data_cluster = []
        count = []
        data_fix_bener = []
        data_fix = []

        for i in range(0, session.get('K')):
            while i is not False: # while 0 == while False, so it just skips to next value
                data_cluster.append(i)
                break 

        for k in range(len(data_cluster)):
            count.insert(k, 0)

        for i in array:
            if i == data_cluster[i]:
                count[i] += 1
                
        data_fix_bener.append(count)
        return data_fix_bener
        # return data_cluster

    # ====== Clustering ======
    @staticmethod
    @app.route('/clustering-baru')
    def clusteringBaru():

        if 'route' in session:
            air = pd.read_csv(session.get('route'))
            dataset = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])

            # Normalisasi
            air_x = air.drop(columns=['Timestamp', 'Nama Lengkap', 'Jenis Kelamin'])
            normalized = preprocessing.normalize(dataset)
            lists = normalized.tolist()

            # Normalisasi MinMax
            x_array = np.array(normalized)
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x_array)
            x_scaled.tolist()

            # Cluster
            if(session.get('K')):
                clustering = int(session.get('K'))

                # Cluster (Cluster 3)
                kmeans = KMeans(n_clusters = clustering, random_state=123)
                kmeans.fit(x_scaled)
                centroid = kmeans.cluster_centers_

                # Cluster
                test = air['kluster'] = kmeans.labels_
                # cluster = 0
                data_fix = 0
                cluster = Clustering.countData(air['kluster'])
                cluster_data = []
                # cluster_data = Clustering.iterate_k_means(x_scaled, centroid, int(session.get('Iteration')))
                
                for i in range(1,int(session.get('K'))):
                    cluster_data.append(Clustering.iterate_k_means_2(x_scaled, centroid, 0))

                graph_test = Clustering.graph()
                graph_2_test = Clustering.graph_2()
                return render_template('clustering/hasil_baru.html', title='Cluster', plot=graph_test, plot2=graph_2_test, cluster_data=cluster_data, table=Clustering.processData(air.values.tolist()), cluster=cluster, data_daerah=data_fix)
                # return jsonify(json.dumps(tests))
                # return render_template('perhitungan/test.html', test=cluster_data, test2=cluster_data)
                # return jsonify(int(session.get('K')))
            else:
                return render_template('clustering/hasil_baru.html', title='Clustering', message='Masukkan Jumlah K terlebih dahulu')
            
        return render_template('clustering/hasil_baru.html', title='Clustering')

    def graph():
        air = pd.read_csv(session.get('route'))
        X = air.loc[:,["Usia", "Pendapatan bulanan", "Jumlah pengeluaran bulanan"]]

        if(session.get('K')):
            _K = session.get('K')
            means_k = KMeans(n_clusters=_K, random_state=0)
            means_k.fit(X)
            labels = means_k.labels_
            centroids = means_k.cluster_centers_

            data = [
                go.Scatter3d(
                    x= X['Jumlah pengeluaran bulanan'],
                    y= X['Pendapatan bulanan'],
                    z= X['Usia'],
                    mode='markers',
                    marker=dict(
                        color = labels, 
                        size= 10,
                        line=dict(
                            color= labels,
                        ),
                        opacity = 0.9
                    )
                )
            ]
            graphJSON = json.dumps(data, cls=py.utils.PlotlyJSONEncoder)

            return graphJSON
        else:
            return False

    def graph_2():
        air = pd.read_csv(session.get('route'))
        X = air.loc[:,["Usia", "Jarak", "Jumlah kunjungan"]]

        if(session.get('K')):
            _K = session.get('K')
            means_k = KMeans(n_clusters=_K, random_state=0)
            means_k.fit(X)
            labels = means_k.labels_
            centroids = means_k.cluster_centers_

            data = [
                go.Scatter3d(
                    x= X['Jumlah kunjungan'],
                    y= X['Jarak'],
                    z= X['Usia'],
                    mode='markers',
                    marker=dict(
                        color = labels, 
                        size= 10,
                        line=dict(
                            color= labels,
                        ),
                        opacity = 0.9
                    )
                )
            ]
            graphJSON = json.dumps(data, cls=py.utils.PlotlyJSONEncoder)

            return graphJSON
        else:
            return False


    def iterate_k_means(data_points, centroids, total_iteration):
        label = []
        cluster_label = []
        list_distance = []
        total_points = len(data_points)
        k = len(centroids)
        new_centroid = []
        
        for iteration in range(0, total_iteration):
            for index_point in range(0, total_points):
                distance = {}
                for index_centroid in range(0, k):
                    distance[index_centroid] = Clustering.compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
                label = Clustering.assign_label_cluster(distance, data_points[index_point], centroids)
                centroids[label[0]] = Clustering.compute_new_centroids(label[1], centroids[label[0]])
                new_centroid.append(Clustering.compute_new_centroids(label[1], centroids[label[0]]))

                if iteration == (total_iteration - 1):
                    cluster_label.append(label)

        return [cluster_label, centroids, new_centroid]
        # return [cluster_label]

    def iterate_k_means_2(data_points, centroids, total_iteration):
        label = []
        cluster_label = []
        total_points = len(data_points)
        k = len(centroids)
        
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = Clustering.compute_euclidean_distance(data_points[index_point], centroids[index_centroid])

            # Menambahkan Posisi Cluster, Data, dan centroid
            label = Clustering.assign_label_cluster(distance, data_points[index_point], centroids)

            # Mengganti Centroid
            centroids[label[0]] = Clustering.compute_new_centroids(label[1], centroids[label[0]])

            cluster_label.append(label)

        # return [cluster_label, centroids]
        return cluster_label 

    def assign_label_cluster(distance, data_point, centroids):
        index_of_minimum = min(distance, key=distance.get)
        return [index_of_minimum, data_point, centroids[index_of_minimum]]

    def compute_new_centroids(cluster_label, centroids):
        return np.array(cluster_label + centroids)/2

    def compute_euclidean_distance(point, centroid):
        return np.sqrt(np.sum((point - centroid)**2))

    def print_label_data(result):
        # print("Result of k-Means Clustering: \n")
        for data in result[0]:
            print("data point: {}".format(data[1]))
            print("cluster number: {} \n".format(data[0]))
        print("Last centroids position: \n {}".format(result[1]))
            






        
        

        



        