from rest_framework.parsers import FileUploadParser, MultiPartParser
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import UploadSerializer
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import action
from drf_yasg import openapi
from rest_framework.status import HTTP_400_BAD_REQUEST
import os
import joblib
import re 
import cv2
import numpy as np
from django.http import JsonResponse
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ultralytics import YOLO
import base64

model = YOLO("D:/102190363/Cap-Project/pbl/uploadimage/best_cccd_ok.pt")
# model = YOLO("D:/102190363/Cap-Project/pbl/uploadimage/best_ylv8cccd.pt") 

model_corner = YOLO("D:/102190363/Cap-Project/pbl/uploadimage/best_corner.pt") 

model_path = r'D:\102190363\Cap-Project\pbl\uploadimage\best_cccd.pt'
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = r'D:\102190363\Cap-Project\pbl\uploadimage\transformerocrr.pth'
config['device'] = 'cpu'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False
detector = Predictor(config)


class UploadFile(APIView):
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
        operation_description='Upload Image...',
        manual_parameters=[
            openapi.Parameter('image', openapi.IN_FORM, type=openapi.TYPE_FILE,
                              description='Image to be uploaded'),
        ]
    )
    @action(detail=True, methods=['POST'])
    def post(self, request):
        serializer = UploadSerializer(data=request.data)


        if not serializer.is_valid():
            return Response(
                {"errors": serializer.errors},
                HTTP_400_BAD_REQUEST
            )

        instance = serializer.save()



        print("Duong dan cua anh: ", instance.image)
        img_url = str(instance.image)

        # Nhan dien
        
        # model_ai = joblib.load('model_pbl')

        img_dir = os.path.dirname(instance.image.path) 
        res_corner_path = os.path.join(img_dir, 'res_corner.jpg')
        res_path = os.path.join(img_dir, 'res.jpg')
        res_box_path = os.path.join(img_dir, 'res_box.jpg')

        # ----------------------------------------------------------------
        # detect 4 gốc, làm phẳng, crop

        image = cv2.imread(img_url)

        results_cn = model_corner.predict(image)
        rect = np.zeros((4,2), dtype='float32')
        acc = 0

        # lap qua cac ket qua
        for result in results_cn:

            boxes = result.boxes.cpu().numpy()

            # lap qua cac bouding box
            for box in boxes:

                x, y, w, h = box.xywh[0].astype(int)      # xywh bbox

                if(result.names[int(box.cls[0])] == 'emblem'):
                    acc = box.conf

                if(result.names[int(box.cls[0])] == 'top_left'):
                    # print('top_left conf',box.conf)
                    if(box.conf > 0.5):
                        rect[0] = x, y
                elif(result.names[int(box.cls[0])] == 'top_right'):
                    # print('top_right conf',box.conf)
                    if(box.conf > 0.5):
                        rect[1] = x, y
                elif(result.names[int(box.cls[0])] == 'bottom_right'):
                    if(box.conf > 0.5):
                        rect[2] = x, y
                elif(result.names[int(box.cls[0])] == 'bottom_left'):
                    if(box.conf > 0.5):
                        rect[3] = x, y

        # hiển thị ảnh predict
        for r in results_cn:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            # im.save(res_corner_path)  # save image


        if(acc > 0.9):
            new_rect = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
            M = cv2.getPerspectiveTransform(rect, new_rect)
            output = cv2.warpPerspective(image, M, (500, 300))

            # cv2.imshow("Crop", output)
            cv2.imwrite(res_corner_path, output)    # save image 'res_corner.jpg'
            # cv2.imwrite(res_corner_path)
            # cv2.waitKey()
            img_crop = cv2.imread(res_corner_path)      # đọc ảnh đã crop
            results = model.predict(img_crop, conf=0.55)    # nhận diện các vùng kí tự

            
            for r in results:
                # im_array = r.plot()
                im_array = r.plot(conf=False, labels=False)  # plot a BGR numpy array of predictions
                # Chuyển sang định dạng BGR -> RGB để hiển thị 
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                # im.show()  # show image
                im.save(res_path)       # lưu ảnh sau khi có các vùng kí tự

            texts = []
            result_dict = {}
            res_sort = ['ID', 'Name', 'Date of birth', 'Sex', 'National', 'Home', 'Address']

            # lặp qua kết quả của các vùng kí tự
            for result in results:
                boxes = result.boxes.cpu().numpy()

                for box in boxes:
                    cls_name = result.names[int(box.cls[0])]

                    x1, y1, x2, y2 = box.xyxy[0].astype(int)

                    # Cắt phần ảnh tương ứng với bbox
                    cropped = img_crop[y1:y2, x1:x2+1]
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    # cv2.imwrite('D:/102190363/Cap-Project/pbl/images/x.jpg', cropped)
                    # imgg = 'D:/102190363/Cap-Project/pbl/images/x.jpg'
                    cv2.imwrite(res_box_path, cropped)
                    imgg = res_box_path
                    img = Image.open(imgg)

                    # print(cls_name + ':' + detector.predict(img, return_prob=False))
                    text = detector.predict(img, return_prob=False)     # sử dụng OCR để chuyển ảnh thành text

                    if(cls_name == 'id'):
                        result_dict['ID'] = text
                    elif(cls_name == 'name'):
                        result_dict['Name'] = text
                    elif(cls_name == 'dob'):
                        result_dict['Date of birth'] = text
                    elif(cls_name == 'sex'):
                        result_dict['Sex'] = text
                    elif(cls_name == 'add1'):
                        result_dict['Address1'] = text
                    elif(cls_name == 'add2'):
                        result_dict['Address2'] = text
                    elif(cls_name == 'home'):
                        result_dict['Home'] = text
                    result_dict['National'] = 'Việt Nam'
                    # print(text)
                    texts.append(text)

                if 'Address1' in result_dict:
                    # Xóa đi dấu phẩy dư thừa ở cuối
                    # result_dict['Address1'] = result_dict['Address1'].rstrip(',')
                    result_dict['Address1'] = re.sub(r'[,.]$', '', result_dict['Address1'])


                if 'Address1' in result_dict and 'Address2' in result_dict:
                    result_dict['Address'] = result_dict['Address1'] + ', ' + result_dict['Address2']
                else: 
                    if 'Address1' in result_dict:
                        result_dict['Address'] = result_dict['Address1']
                    if 'Address2' in result_dict:
                        result_dict['Address'] = result_dict['Address2']
                
                # in kq ra terminal
                for key, value in result_dict.items():
                    if isinstance(value, list):
                        print(f"{key}:")
                        for item in value:
                            print(f"   {item}")
                    else:
                        print(f"{key}: {value}")
            # Xoa anh sau khi nhan dien
            os.remove(str(instance.image))

            if(len(texts) < 4):
                    return Response(status=400)
            # sắp xếp theo thứ tự
            sorted_result_dict = {key: result_dict[key].upper() for key in res_sort if key in result_dict}

            # chuyển ảnh về base64
            with open(res_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            response_data = {
                'result': sorted_result_dict,
                'image': f"data:image/jpeg;base64,{encoded_image}"
            }
            return JsonResponse(response_data)
        else:
            os.remove(str(instance.image))
            print("xin chụp lại ảnh")
            return Response(status=400)

        # ----------------------------------------------------------------
        
        
        # detect các vùng trên cccd

        # img_crop = cv2.imread(res_corner_path)
        # results = model.predict(img_crop, conf=0.55)

        # # hiển thị ảnh predict
        # for r in results:
        #     # im_array = r.plot()
        #     im_array = r.plot(conf=False, labels=False)  # plot a BGR numpy array of predictions
        #     # Chuyển sang định dạng BGR -> RGB để hiển thị 
        #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #     # im.show()  # show image
        #     im.save(res_path)  # save image

        # texts = []
        # result_dict = {}
        # desired_order = ['ID', 'Name', 'Date of birth', 'Sex', 'National', 'Address', 'Home']
        # for result in results:
        #     boxes = result.boxes.cpu().numpy()

        #     for box in boxes:
        #         cls_name = result.names[int(box.cls[0])]

        #         x1, y1, x2, y2 = box.xyxy[0].astype(int)

        #         # Cắt phần ảnh tương ứng với bbox
        #         cropped = img_crop[y1:y2, x1:x2]
        #         # cv2.imwrite('D:/102190363/Cap-Project/pbl/images/x.jpg', cropped)
        #         # imgg = 'D:/102190363/Cap-Project/pbl/images/x.jpg'
        #         cv2.imwrite(res_box_path, cropped)
        #         imgg = res_box_path
        #         img = Image.open(imgg)
        #         # img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        #         # print(cls_name + ':' + detector.predict(img, return_prob=False))
        #         text = detector.predict(img, return_prob=False)

        #         if(cls_name == 'id'):
        #             result_dict['ID'] = text
        #         elif(cls_name == 'name'):
        #             result_dict['Name'] = text
        #         elif(cls_name == 'dob'):
        #             result_dict['Date of birth'] = text
        #         elif(cls_name == 'sex'):
        #             result_dict['Sex'] = text
        #         elif(cls_name == 'add'):
        #             result_dict['Address'] = text
        #         elif(cls_name == 'home'):
        #             result_dict['Home'] = text
        #         result_dict['National'] = 'Việt Nam'
        #         # print(text)
        #         texts.append(text)
        #     for key, value in result_dict.items():
        #         if isinstance(value, list):
        #             print(f"{key}:")
        #             for item in value:
        #                 print(f"   {item}")
        #         else:
        #             print(f"{key}: {value}")
            
        # # Xoa anh sau khi nhan dien
        # os.remove(str(instance.image))
        # sorted_result_dict = {key: result_dict[key] for key in desired_order if key in result_dict}

        # with open(res_path, "rb") as image_file:
        #     encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # response_data = {
        #     'result': sorted_result_dict,
        #     'image': f"data:image/jpeg;base64,{encoded_image}"
        # }
        # return JsonResponse(response_data)
        # return JsonResponse(sorted_result_dict)
        # return instance.image

