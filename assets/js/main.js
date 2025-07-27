/**
 * Template Name: Scout
 * Template URL: https://bootstrapmade.com/scout-bootstrap-multipurpose-template/
 * Updated: May 05 2025 with Bootstrap v5.3.5
 * Author: BootstrapMade.com
 * License: https://bootstrapmade.com/license/
 */
const API_BASE =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "https://admittedly-rich-crow.ngrok-free.app";

(function () {
  "use strict";

  /**
   * Apply .scrolled class to the body as the page is scrolled down
   */
  function toggleScrolled() {
    const selectBody = document.querySelector("body");
    const selectHeader = document.querySelector("#header");
    if (
      !selectHeader.classList.contains("scroll-up-sticky") &&
      !selectHeader.classList.contains("sticky-top") &&
      !selectHeader.classList.contains("fixed-top")
    )
      return;
    window.scrollY > 100
      ? selectBody.classList.add("scrolled")
      : selectBody.classList.remove("scrolled");
  }

  document.addEventListener("scroll", toggleScrolled);
  window.addEventListener("load", toggleScrolled);

  /**
   * Mobile nav toggle
   */
  const mobileNavToggleBtn = document.querySelector(".mobile-nav-toggle");

  function mobileNavToogle() {
    document.querySelector("body").classList.toggle("mobile-nav-active");
    mobileNavToggleBtn.classList.toggle("bi-list");
    mobileNavToggleBtn.classList.toggle("bi-x");
  }
  if (mobileNavToggleBtn) {
    mobileNavToggleBtn.addEventListener("click", mobileNavToogle);
  }

  /**
   * Hide mobile nav on same-page/hash links
   */
  document.querySelectorAll("#navmenu a").forEach((navmenu) => {
    navmenu.addEventListener("click", () => {
      if (document.querySelector(".mobile-nav-active")) {
        mobileNavToogle();
      }
    });
  });

  /**
   * Toggle mobile nav dropdowns
   */
  document.querySelectorAll(".navmenu .toggle-dropdown").forEach((navmenu) => {
    navmenu.addEventListener("click", function (e) {
      e.preventDefault();
      this.parentNode.classList.toggle("active");
      this.parentNode.nextElementSibling.classList.toggle("dropdown-active");
      e.stopImmediatePropagation();
    });
  });

  /**
   * Scroll top button
   */
  let scrollTop = document.querySelector(".scroll-top");

  function toggleScrollTop() {
    if (scrollTop) {
      window.scrollY > 100
        ? scrollTop.classList.add("active")
        : scrollTop.classList.remove("active");
    }
  }
  scrollTop.addEventListener("click", (e) => {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });

  window.addEventListener("load", toggleScrollTop);
  document.addEventListener("scroll", toggleScrollTop);

  /**
   * Animation on scroll function and init
   */
  function aosInit() {
    AOS.init({
      duration: 600,
      easing: "ease-in-out",
      once: true,
      mirror: false,
    });
  }
  window.addEventListener("load", aosInit);

  /**
   * Initiate Pure Counter
   */
  new PureCounter();

  /**
   * Frequently Asked Questions Toggle
   */
  document
    .querySelectorAll(".faq-item h3, .faq-item .faq-toggle")
    .forEach((faqItem) => {
      faqItem.addEventListener("click", () => {
        faqItem.parentNode.classList.toggle("faq-active");
      });
    });

  /**
   * Init swiper sliders
   */
  function initSwiper() {
    document.querySelectorAll(".init-swiper").forEach(function (swiperElement) {
      let config = JSON.parse(
        swiperElement.querySelector(".swiper-config").innerHTML.trim()
      );

      if (swiperElement.classList.contains("swiper-tab")) {
        initSwiperWithCustomPagination(swiperElement, config);
      } else {
        new Swiper(swiperElement, config);
      }
    });
  }

  window.addEventListener("load", initSwiper);

  /**
   * Correct scrolling position upon page load for URLs containing hash links.
   */
  window.addEventListener("load", function (e) {
    if (window.location.hash) {
      if (document.querySelector(window.location.hash)) {
        setTimeout(() => {
          let section = document.querySelector(window.location.hash);
          let scrollMarginTop = getComputedStyle(section).scrollMarginTop;
          window.scrollTo({
            top: section.offsetTop - parseInt(scrollMarginTop),
            behavior: "smooth",
          });
        }, 100);
      }
    }
  });

  /**
   * Navmenu Scrollspy
   */
  let navmenulinks = document.querySelectorAll(".navmenu a");

  function navmenuScrollspy() {
    navmenulinks.forEach((navmenulink) => {
      if (!navmenulink.hash) return;
      let section = document.querySelector(navmenulink.hash);
      if (!section) return;
      let position = window.scrollY + 200;
      if (
        position >= section.offsetTop &&
        position <= section.offsetTop + section.offsetHeight
      ) {
        document
          .querySelectorAll(".navmenu a.active")
          .forEach((link) => link.classList.remove("active"));
        navmenulink.classList.add("active");
      } else {
        navmenulink.classList.remove("active");
      }
    });
  }
  window.addEventListener("load", navmenuScrollspy);
  document.addEventListener("scroll", navmenuScrollspy);
})();

// Simulasikan generate gambar (pakai URL dummy)
function generateDummyImages(selectedClasses, count) {
  const images = {};
  selectedClasses.forEach((cls) => {
    images[cls] = [];
    for (let i = 0; i < count; i++) {
      const dummyUrl = `https://via.placeholder.com/128.png?text=Class+${cls}+Img+${
        i + 1
      }`;
      images[cls].push(dummyUrl);
    }
  });
  return images;
}

function renderPreview(imagesByClass, selectedClasses) {
  const container = document.getElementById("previewContainer");
  container.innerHTML = ""; // Kosongkan preview

  selectedClasses.forEach(({ value, label }) => {
    const images = imagesByClass[value];
    const col = document.createElement("div");
    col.className = "col-12";

    const card = document.createElement("div");
    card.className = "card shadow-sm mb-4";

    const cardBody = document.createElement("div");
    cardBody.className = "card-body";

    const title = document.createElement("h5");
    title.textContent = `${label} (Class ${value})`;
    title.classList.add("card-title");

    const carouselId = `carousel-class${value}`;
    const carousel = document.createElement("div");
    carousel.className = "carousel slide";
    carousel.id = carouselId;
    carousel.setAttribute("data-bs-ride", "carousel");

    const carouselInner = document.createElement("div");
    carouselInner.className = "carousel-inner";

    // Split images into chunks of 5
    const chunkSize = 5;
    for (let i = 0; i < images.length; i += chunkSize) {
      const chunk = images.slice(i, i + chunkSize);
      const item = document.createElement("div");
      item.className = `carousel-item ${i === 0 ? "active" : ""}`;

      const row = document.createElement("div");
      row.className = "row justify-content-center align-items-center gx-3";

      chunk.forEach((imgSrc) => {
        const imgCol = document.createElement("div");
        imgCol.className =
          "col-6 col-md-3 col-lg-2 d-flex justify-content-center align-items-center mb-3";

        imgCol.innerHTML = `
      <img src="${imgSrc}" class="rounded border mx-auto" alt="Generated image" 
        style="width: 128px; height: 128px; object-fit: cover;">
    `;
        row.appendChild(imgCol);
      });

      item.appendChild(row);
      carouselInner.appendChild(item);
    }

    const prevBtn = `
      <button class="carousel-control-prev" type="button" data-bs-target="#${carouselId}" data-bs-slide="prev">
        <span class="carousel-control-prev-icon"></span>
        <span class="visually-hidden">Previous</span>
      </button>`;
    const nextBtn = `
      <button class="carousel-control-next" type="button" data-bs-target="#${carouselId}" data-bs-slide="next">
        <span class="carousel-control-next-icon"></span>
        <span class="visually-hidden">Next</span>
      </button>`;

    carousel.appendChild(carouselInner);
    carousel.insertAdjacentHTML("beforeend", prevBtn + nextBtn);

    const countInfo = document.createElement("p");
    countInfo.className = "text-muted small mt-2";
    countInfo.textContent = `${images.length} gambar dihasilkan`;

    cardBody.appendChild(title);
    cardBody.appendChild(carousel);
    cardBody.appendChild(countInfo);
    card.appendChild(cardBody);
    col.appendChild(card);
    container.appendChild(col);
  });
}

// Kelengkapan untuk Generate Model tujuan1.html

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("generateForm");
  const downloadBtn = document.getElementById("downloadBtn");
  const resetBtn = document.getElementById("resetBtn");
  const previewContainer = document.getElementById("previewContainer");

  const loadingSpinner = document.createElement("div");
  loadingSpinner.className = "text-center my-4";
  loadingSpinner.innerHTML = `
    <div class="spinner-border text-primary" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
    <p>Memproses gambar sintetis, mohon tunggu...</p>
  `;

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    const model = document.getElementById("modelSelect").value;
    const imageCount = parseInt(document.getElementById("imageCount").value);

    const selectedClasses = Array.from(
      document.querySelectorAll('input[type="checkbox"]:checked')
    ).map((input) => ({
      value: input.value,
      label: input.dataset.label,
    }));

    if (
      !model ||
      selectedClasses.length === 0 ||
      isNaN(imageCount) ||
      imageCount < 1 ||
      imageCount > 100
    ) {
      alert("Lengkapi semua isian dengan benar.");
      return;
    }

    previewContainer.innerHTML = "";
    previewContainer.appendChild(loadingSpinner);
    downloadBtn.disabled = true;

    fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        image_count: imageCount,
        classes: selectedClasses.map((c) => parseInt(c.value)),
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        renderCarousel(data.generated, selectedClasses);
        downloadBtn.disabled = false;
      })
      .catch((err) => {
        console.error("Gagal generate gambar:", err);
        alert("Terjadi kesalahan saat menghubungi server.");
      });
  });

  resetBtn.addEventListener("click", function () {
    previewContainer.innerHTML = "";
    downloadBtn.disabled = true;
  });

  downloadBtn.addEventListener("click", function () {
    fetch(`${API_BASE}/download`, {
      headers: {
        "ngrok-skip-browser-warning": "true",
      },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Gagal download ZIP");
        return res.blob();
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "generated_images.zip";
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      })
      .catch((err) => {
        alert("Gagal mengunduh ZIP: " + err.message);
      });
  });

  // Update ikon centang checkbox
  const allCheckboxes = document.querySelectorAll(
    'input[type="checkbox"][id^="class"]'
  );

  allCheckboxes.forEach((checkbox) => {
    const icon = document.querySelector(
      `.icon-check[data-id="${checkbox.value}"]`
    );

    updateIcon(checkbox, icon); // Initial state

    checkbox.addEventListener("change", () => {
      updateIcon(checkbox, icon);
    });
  });

  function updateIcon(checkbox, icon) {
    if (checkbox.checked) {
      icon.classList.remove("fa-square");
      icon.classList.add("fa-check-square", "active");
    } else {
      icon.classList.remove("fa-check-square", "active");
      icon.classList.add("fa-square");
    }
  }

  // Tampilkan form saat tombol diklik
  const showFormBtn = document.getElementById("showFormBtn");
  const cgmFormContainer = document.getElementById("cgmFormContainer");

  showFormBtn.addEventListener("click", function () {
    cgmFormContainer.classList.remove("d-none");
    showFormBtn.classList.add("d-none");
    window.scrollTo({
      top: cgmFormContainer.offsetTop - 60,
      behavior: "smooth",
    });
  });

  function renderCarousel(imagesByClass, selectedClasses) {
    previewContainer.innerHTML = "";
    previewContainer.appendChild(loadingSpinner); // Tambahkan spinner di awal

    const renderTasks = selectedClasses.map(({ value, label }) => {
      const images = imagesByClass[value] || [];
      const carouselId = `carousel-${value}`;
      const indicatorId = `carousel-indicators-${value}`;
      const col = document.createElement("div");
      col.className = "col-12 mb-5";

      const box = document.createElement("div");
      box.className = "carousel-box";

      const title = document.createElement("h5");
      title.textContent = `Kelas ${label}`;
      box.appendChild(title);

      const groupedImages = images.reduce((chunks, url, i) => {
        if (i % 6 === 0) chunks.push([]);
        chunks[chunks.length - 1].push(url);
        return chunks;
      }, []);

      const carouselInnerPromises = groupedImages.map((group, idx) =>
        Promise.all(
          group.map((url) => {
            const isFullURL =
              url.startsWith("http://") || url.startsWith("https://");
            const isLocalhost =
              url.includes("localhost") || url.includes("127.0.0.1");
            const fullURL =
              isFullURL && !isLocalhost
                ? url
                : `${API_BASE}${url.startsWith("/") ? "" : "/"}${url}`;

            return fetch(fullURL, {
              headers: { "ngrok-skip-browser-warning": "true" },
            })
              .then((res) => res.blob())
              .then((blob) => {
                const imgURL = URL.createObjectURL(blob);
                return `<img src="${imgURL}" class="img-thumbnail" style="width: 128px; height: 128px;" loading="lazy">`;
              })
              .catch(() => `<div class="text-danger">Gagal load</div>`);
          })
        ).then((imageElements) => {
          return `
          <div class="carousel-item ${idx === 0 ? "active" : ""}">
            <div class="d-flex flex-wrap justify-content-center gap-2">
              ${imageElements.join("")}
            </div>
          </div>
        `;
        })
      );

      return Promise.all(carouselInnerPromises).then((carouselItemsHTML) => {
        const indicators = groupedImages
          .map((_, i) => {
            return `<button type="button" data-bs-target="#${carouselId}" data-bs-slide-to="${i}" class="${
              i === 0 ? "active" : ""
            }" aria-current="${i === 0 ? "true" : "false"}" aria-label="Slide ${
              i + 1
            }"></button>`;
          })
          .join("");

        const carousel = `
        <div id="${carouselId}" class="carousel slide" data-bs-ride="carousel">
          <div class="carousel-indicators">${indicators}</div>
          <div class="carousel-inner">${carouselItemsHTML.join("")}</div>
          <button class="carousel-control-prev" type="button" data-bs-target="#${carouselId}" data-bs-slide="prev">
            <span class="carousel-control-prev-icon"></span>
          </button>
          <button class="carousel-control-next" type="button" data-bs-target="#${carouselId}" data-bs-slide="next">
            <span class="carousel-control-next-icon"></span>
          </button>
        </div>
      `;

        box.innerHTML += carousel;
        col.appendChild(box);
        previewContainer.appendChild(col);
      });
    });

    Promise.all(renderTasks).then(() => {
      const spinner = document.querySelector(".spinner-border")?.parentElement;
      if (spinner) spinner.remove();
    });
  }
});

// Kelengkapan JS untuk Klasifikasi Citra tujuan3.html

document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageUpload");
  const previewImage = document.getElementById("previewImage");
  const resultContainer = document.getElementById("classificationResult");
  const resultText = document.getElementById("resultText");
  const classifyBtn = document.getElementById("classifyBtn");
  const switchCameraBtn = document.getElementById("switchCameraBtn");

  let currentFacingMode = "environment";
  let stream;

  if (imageInput) {
    imageInput.addEventListener("change", function () {
      const file = this.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          previewImage.src = e.target.result;
          previewImage.classList.remove("d-none");
          classifyBtn.classList.remove("d-none");
          resultContainer.classList.add("d-none");
        };
        reader.readAsDataURL(file);
      }
    });
  }

  window.openCamera = function () {
    const video = document.getElementById("cameraPreview");
    const cameraSection = document.getElementById("cameraSection");
    cameraSection.classList.remove("d-none");

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: currentFacingMode },
        audio: false,
      })
      .then((newStream) => {
        stream = newStream;
        video.srcObject = stream;
      })
      .catch((err) => {
        console.error(err);
        alert("Tidak dapat mengakses kamera.");
      });
  };

  window.switchCamera = function () {
    currentFacingMode =
      currentFacingMode === "environment" ? "user" : "environment";
    window.openCamera();
  };

  window.captureImage = function () {
    const video = document.getElementById("cameraPreview");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL("image/png");

    previewImage.src = dataURL;
    previewImage.classList.remove("d-none");
    classifyBtn.classList.remove("d-none");
    resultContainer.classList.add("d-none");
  };

  if (classifyBtn) {
    document.getElementById("explainResult")?.classList.add("d-none");
    classifyBtn.addEventListener("click", async function () {
      resultText.textContent = "⏳ Memproses...";
      resultContainer.classList.remove("d-none");

      const fileInput = imageInput.files[0];
      let file;

      if (fileInput) {
        file = fileInput;
      } else {
        const dataURL = previewImage.src;
        const blob = await (await fetch(dataURL)).blob();
        file = new File([blob], "capture.png", { type: "image/png" });
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          body: formData,
        });
        const data = await res.json();

        if (data.error) {
          resultText.textContent = `❌ Error: ${data.error}`;
          resultContainer.classList.replace("alert-info", "alert-danger");
        } else {
          resultText.textContent = `✅ ${data.label} (${(
            data.confidence * 100
          ).toFixed(2)}%)`;
          resultContainer.classList.replace("alert-danger", "alert-info");

          // ✅ Tampilkan tombol interpretasi setelah sukses klasifikasi
          if (explainBtn) {
            explainBtn.classList.remove("d-none");
          }
        }
      } catch (err) {
        resultText.textContent = "❌ Gagal terhubung ke server.";
        resultContainer.classList.replace("alert-info", "alert-danger");
      }
    });
  }

  // Sembunyikan tombol switch kamera di desktop
  const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  if (!isMobile && switchCameraBtn) {
    switchCameraBtn.style.display = "none";
  }
});

// Fungsi Interpretasi Citra yang Diklasifikasikan
window.requestExplain = async function () {
  const imageInput = document.getElementById("imageUpload");
  const previewImage = document.getElementById("previewImage");

  const fileInput = imageInput.files[0];
  let file;

  if (fileInput) {
    file = fileInput;
  } else {
    const dataURL = previewImage.src;
    const blob = await (await fetch(dataURL)).blob();
    file = new File([blob], "capture.png", { type: "image/png" });
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/predict?with_explain=true`, {
      method: "POST",
      body: formData,
    });

    const text = await res.text(); // Ambil response mentah
    console.log("Respon Mentah:", text);

    const data = JSON.parse(text); // Parse manual
    console.log("Parsed JSON:", data);

    if (data.explanation_image) {
      document.getElementById("gradcamImage").src =
        "data:image/png;base64," + data.explanation_image;

      document.getElementById("originalImage").src =
        document.getElementById("previewImage").src;

      document.getElementById("explainResult").classList.remove("d-none");
    } else {
      alert("❌ Tidak ada gambar Grad-CAM yang diterima.");
    }
  } catch (err) {
    console.error("Gagal memuat Grad-CAM:", err);
    alert("❌ Gagal menghubungi server untuk interpretasi.");
  }
};
