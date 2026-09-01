# Unified Day–Night Dehazing (Mixture of Experts)

Project tích hợp **bộ định tuyến ngày/đêm**, DehazeNet cho ảnh ban ngày và
Guided APSF / Gradient-Adaptive CNN cho ảnh ban đêm. Hai repository gốc vẫn là
các dependency độc lập; project không chép lại mã hoặc checkpoint của tác giả.

![Workflow](docs/workflow.png)

## Vì sao kiến trúc này hợp lý?

Không nên nối DehazeNet sau nighttime_dehaze: một model có thể phá màu hoặc chi
tiết mà model sau cần. Thay vào đó, đây là **mixture of experts**:

1. `SceneRouter` đọc ảnh thu nhỏ và tạo `p_night` từ độ sáng, phân vị vùng tối,
   độ bão hòa và tỉ lệ highlight. Bộ router mặc định không cần checkpoint.
2. Nếu router chắc chắn, hệ thống chỉ chạy expert phù hợp để giảm thời gian.
3. Nếu ảnh ở chạng vạng/vùng bất định, cả hai expert chạy và `ConfidenceFusion`
   trộn kết quả bằng trọng số mềm. Trọng số được làm trơn để tránh đường biên.
4. `QualityGuard` giới hạn độ lệch sáng/màu quá lớn và loại NaN/giá trị ngoài
   miền. Đây là guardrail, không thay thế việc huấn luyện/evaluate.

### Luồng của Day Expert

DehazeNet ước lượng transmission `t(x)` qua Feature Extraction (Maxout),
Multi-scale Mapping, Local Extremum và BReLU. Atmospheric light `A` cùng `t(x)`
được dùng trong `J(x) = (I(x)-A)/max(t(x), eps) + A`. Adapter gọi trực tiếp
`run_cnn.m` và `dehaze.mat` từ repository gốc.

### Luồng của Night Expert

Repository nighttime_dehaze dùng generator residual với CAM attention,
ResNet blocks và AdaILN/ILN. Guided APSF được dùng để tạo glow/haze phù hợp ảnh
đêm trong quá trình xây dữ liệu; gradient-adaptive design chú ý cấu trúc/biên và
nguồn sáng. Adapter nạp `ResnetGenerator` và checkpoint `.pt` gốc.

> Lưu ý: APSF không phải một module bắt buộc chạy riêng trước generator ở mọi
> lần inference. Trong repository, phần `APSF_GLOW_RENDER_CODE` chủ yếu tạo cặp
> dữ liệu glow; đừng mô tả sai thành pipeline tuần tự cố định.

## Cài đặt

```bash
git clone https://github.com/caibolun/DehazeNet.git third_party/DehazeNet
git clone https://github.com/jinyeying/nighttime_dehaze.git third_party/nighttime_dehaze
python -m pip install -r requirements.txt
```

Day expert cần MATLAB + MATLAB Engine for Python. Night expert cần tải checkpoint
theo README của repository gốc, ví dụ `dehaze.pt`. Kiểm tra điều kiện giấy phép
trước sử dụng thương mại.

## Chạy

```bash
python -m unified_dehaze.cli input.jpg output.png \
  --dehazenet-root third_party/DehazeNet \
  --night-root third_party/nighttime_dehaze \
  --night-checkpoint weights/dehaze.pt \
  --mode auto --device cuda
```

Các mode: `auto`, `day`, `night`. `auto` dùng routing; `day/night` hữu ích để
ablation test. Metadata routing được ghi cạnh output dưới dạng JSON nếu thêm
`--metadata result.json`.

## Tối ưu đề xuất

- **Giai đoạn 1:** dùng heuristic router hiện tại để có baseline dễ giải thích.
- **Giai đoạn 2:** huấn luyện MobileNetV3-Small/ConvNeXt-Tiny classifier bằng
  nhãn `day, twilight, night`; hiệu chỉnh xác suất bằng temperature scaling.
- **Giai đoạn 3:** fine-tune router theo chất lượng expert: nhãn tốt nhất là
  expert có LPIPS thấp nhất/PSNR cao nhất, không chỉ nhãn thời gian trong ngày.
- **Đánh giá riêng:** Day (RESIDE), Night (NHR/NHM/NHC/GTA5) và Twilight; báo cáo
  PSNR, SSIM, LPIPS, thời gian và lỗi routing.

## Nguồn

- DehazeNet: https://github.com/caibolun/DehazeNet
- Nighttime dehaze: https://github.com/jinyeying/nighttime_dehaze
- DehazeNet paper (TIP 2016); Guided APSF paper (ACM MM 2023).

