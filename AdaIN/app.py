import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from train import VGGEncoder, Decoder, adaptive_instance_normalization

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdaINStyleTransfer:
    def __init__(self, model_path='model_save/adain_model.pth'):
        self.device = device
        self.model_path = model_path

        # 모델 초기화
        self.encoder = VGGEncoder().to(self.device)
        self.decoder = Decoder().to(self.device)

        # 사전 훈련된 모델 로드
        self.load_model()

        # 평가 모드로 설정
        self.encoder.eval()
        self.decoder.eval()

        # 정규화 해제용 변환
        self.denormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

    def load_model(self):
        """사전 훈련된 모델 로드"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Decoder 가중치 로드
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

            print(f"Model loaded successfully from {self.model_path}")

            # 손실 정보가 있다면 출력
            if 'losses' in checkpoint:
                final_loss = checkpoint['losses'][-1]
                print(f"Final training loss: {final_loss:.4f}")

        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def resize_keep_aspect_ratio(self, image, target_size=512):
        """비율을 유지하면서 이미지 리사이즈"""
        original_width, original_height = image.size

        # 긴 쪽을 target_size에 맞추기
        if original_width > original_height:
            new_width = target_size
            new_height = int((original_height * target_size) / original_width)
        else:
            new_height = target_size
            new_width = int((original_width * target_size) / original_height)

        # 최소 크기 보장 (너무 작으면 품질이 떨어질 수 있음)
        new_width = max(new_width, 256)
        new_height = max(new_height, 256)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def load_image(self, image_path, target_size=512):
        """이미지 로드 및 전처리"""
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (width, height)

            # 비율을 유지하면서 리사이즈
            resized_image = self.resize_keep_aspect_ratio(image, target_size)

            # 정규화
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            image_tensor = transform(resized_image).unsqueeze(0)  # 배치 차원 추가
            return image_tensor.to(self.device), original_size, resized_image.size

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

    def save_image(self, tensor, save_path, target_size):
        """텐서를 이미지로 저장"""
        # 정규화 해제
        tensor = self.denormalize(tensor.squeeze(0).cpu())

        # 값 범위를 [0, 1]로 클램핑
        tensor = torch.clamp(tensor, 0, 1)

        # PIL 이미지로 변환
        to_pil = transforms.ToPILImage()
        image = to_pil(tensor)

        # 원본 크기로 복원
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        # 저장
        image.save(save_path, quality=95)  # 고품질로 저장
        print(f"Result saved to: {save_path} (size: {target_size})")

    def style_transfer(self, content_path, style_path, alpha=1.0):
        """스타일 전송 수행"""
        with torch.no_grad():
            # 이미지 로드 (콘텐츠 이미지의 원본 크기 저장)
            content_img, original_content_size, processed_content_size = self.load_image(content_path)
            style_img, _, _ = self.load_image(style_path)

            print(
                f"Content image: {content_path} (original: {original_content_size}, processed: {processed_content_size})")
            print(f"Style image: {style_path}")
            print(f"Alpha (style strength): {alpha}")

            # 스타일 이미지가 콘텐츠 이미지보다 작으면 업샘플링
            if style_img.shape[2] < content_img.shape[2] or style_img.shape[3] < content_img.shape[3]:
                style_img = torch.nn.functional.interpolate(
                    style_img,
                    size=(content_img.shape[2], content_img.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )

            # 특징 추출
            content_features = self.encoder(content_img)
            style_features = self.encoder(style_img)

            # AdaIN 적용
            target_features = adaptive_instance_normalization(content_features, style_features)

            # Alpha 블렌딩 (스타일 강도 조절)
            target_features = alpha * target_features + (1 - alpha) * content_features

            # 이미지 생성
            generated_img = self.decoder(target_features)

            return generated_img, original_content_size


def get_available_images(directory):
    """디렉토리에서 사용 가능한 이미지 파일 목록 반환"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []

    if os.path.exists(directory):
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images.append(file)

    return sorted(images)


def interactive_mode():
    """대화형 모드"""
    print("=== AdaIN Style Transfer (Aspect Ratio Preserved) ===")

    # 출력 디렉토리 확인/생성
    os.makedirs('output', exist_ok=True)

    # 스타일 전송 객체 생성
    style_transfer = AdaINStyleTransfer()

    while True:
        print("\n--- Available Images ---")

        # 콘텐츠 이미지 목록
        content_images = get_available_images('input')
        if not content_images:
            print("No images found in 'input' directory!")
            return

        print("Content images in 'input' directory:")
        for i, img in enumerate(content_images):
            print(f"  {i}: {img}")

        try:
            # 콘텐츠 이미지 선택
            content_idx = int(input(f"\nSelect content image (0-{len(content_images) - 1}): "))
            if content_idx < 0 or content_idx >= len(content_images):
                print("Invalid selection!")
                continue

            content_path = os.path.join('input', content_images[content_idx])

            # 스타일 이미지 선택 (input 디렉토리에서)
            print(f"\nSelect style image (0-{len(content_images) - 1}): ")
            style_idx = int(input())
            if style_idx < 0 or style_idx >= len(content_images):
                print("Invalid selection!")
                continue

            style_path = os.path.join('input', content_images[style_idx])

            # 스타일 강도 설정
            alpha = float(input("Enter style strength (0.0-1.0, default 1.0): ") or "1.0")
            alpha = max(0.0, min(1.0, alpha))

            # 스타일 전송 수행
            print("\nProcessing...")
            result, original_size = style_transfer.style_transfer(content_path, style_path, alpha)

            # 결과 저장 (원본 크기로)
            content_name = os.path.splitext(content_images[content_idx])[0]
            style_name = os.path.splitext(content_images[style_idx])[0]
            output_name = f"{content_name}_stylized_by_{style_name}_alpha{alpha:.1f}.jpg"
            output_path = os.path.join('output', output_name)

            style_transfer.save_image(result, output_path, original_size)

            # 계속 여부 확인
            continue_choice = input("\nContinue? (y/n): ").lower()
            if continue_choice != 'y':
                break

        except ValueError:
            print("Please enter a valid number!")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(content_path, style_path, output_path, alpha=1.0):
    """배치 모드 (명령행 인자 사용)"""
    # 출력 디렉토리 확인/생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 스타일 전송 수행
    style_transfer = AdaINStyleTransfer()
    result, original_size = style_transfer.style_transfer(content_path, style_path, alpha)
    style_transfer.save_image(result, output_path, original_size)


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer with Aspect Ratio Preservation')
    parser.add_argument('--content', type=str, help='Path to content image')
    parser.add_argument('--style', type=str, help='Path to style image')
    parser.add_argument('--output', type=str, help='Path to output image')
    parser.add_argument('--alpha', type=float, default=1.0, help='Style strength (0.0-1.0)')

    args = parser.parse_args()

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.content and args.style and args.output:
        # 배치 모드
        batch_mode(args.content, args.style, args.output, args.alpha)
    else:
        # 대화형 모드
        interactive_mode()


if __name__ == '__main__':
    main()