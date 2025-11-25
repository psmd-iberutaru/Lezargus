; This code is from Mike Connelley, with some minor modifications 
; to print out some needed outputs.


device, decomposed=0
TVLCT, 255,   0,   0, 1   ; red
TVLCT, 255, 120,   0, 2   ; orange
TVLCT, 255, 255,   0, 3   ; yellow
; TVLCT, 120, 60,   0, 3   ; brown
TVLCT,   0, 200,   0, 4   ; green
TVLCT,   0, 255, 150, 5   ; green-blue
TVLCT,   0, 255, 255, 6   ; blue-green
TVLCT,  100,  100, 255, 7   ; blue
TVLCT, 150,   0, 255, 8   ; indigo
TVLCT, 255,   0, 255, 9   ; violet
TVLCT, 200, 200, 200, 10   ; gray

; set_plot, 'ps'
; device, filename='/Users/msc/irtf/spectre/Resolution_plots/Resolution_231115.ps', Xsize=8.0, Ysize=6.0, Xoffset=0.5, Yoffset=2.5, /Inches, /encapsulated, color=1

 t=1.
 
; ******************************************
;        NIR   230825 f/12 2 pixels/slice  
; ******************************************
w1 =      [0.8500, 1.000, 1.2500, 1.600, 2.000, 2.400]
; w18 =     [0.8527, 1.004, 1.256, 1.607, 2.007, 2.406]    ; 231114 design w/ 18 deg silica prism angle, 2.5 ZnSe angle
; w16 =     [0.8527, 1.004, 1.2564, 1.608, 2.008, 2.407]    ; 231114 design w/ 16 deg silica prism angle, 2.5 ZnSe angle
; w15 =     [0.8527, 1.004, 1.2564, 1.6082, 2.0082, 2.4073]    ; 231114 design w/ 15 deg silica prism angle, 2.5 ZnSe angle
w16_2 = [0.853, 1.0043, 1.2566, 1.608, 2.008, 2.4065]    ; 231115_prep design w/ 16 deg silica prism angle, 2.25 ZnSe angle, does not consider diffraction or spot size

slit_image = [36., 36., 36., 36., 36., 36.]   ; slit image width in microns
spot = [12., 12., 12., 12., 12., 12]             ; median spot size in microns
diffract = w1*12.   ; diffraction spot size in microns

blur = sqrt(slit_image^2 + spot^2 + diffract^2)
degrade = blur/slit_image

print, diffract
print, blur
print, degrade

; R18 = w1/(w18-w1)            ; w/ 18 deg wedge, spectrum is 9.54 mm long
; R16 = w1/(w16-w1)            ; w/ 16 deg wedge, spectrum is 8.81 mm long
; R15 = w1/(w15-w1)            ; w/ 16 deg wedge, spectrum is 8.43 mm long
R16_2 = w1/(w16_2-w1)            ; w/ 16 deg wedge, spectrum is 8.49 mm long

plot, w1, r16_2, xtitle='Wavelength (microns)', ytitle='Spectral Resolution', xstyle=1, xrange=[0.35, 5], ystyle=1, yrange=[00, 750], psym=-1, /xlog, thick=t, xthick=t, ythick=t, charthick=t
; oplot, w1, r16, color=1
; oplot, w1, r16_2, color=2

oplot, w1, r16_2/degrade, linestyle=2, psym=-1, thick=t

; print, 'Minimum NIR resolution:', min(r18),  '       Mean NIR resolution:', mean(r18)
; print, 'Minimum NIR resolution:', min(r16),  '       Mean NIR resolution:', mean(r16)
print, 'Minimum NIR resolution:', min(r16_2),  '       Mean NIR resolution:', mean(r16_2)
print, 'Minimum NIR resolution:', min(r16_2/degrade),  '       Mean NIR resolution:', mean(r16_2/degrade)
; print, 'Minimum NIR resolution:', min(r18)/1.11,  '       Mean NIR resolution:', mean(r18)/1.11
; print, 'Minimum NIR resolution:', min(r16)/1.11,  '       Mean NIR resolution:', mean(r16)/1.11

print, "NIR"
print, "Wavelength", w1 
print, "Resolution", r16_2
print, "Degraded  ", r16_2/degrade

; ******************************************
;        Thermal-IR
; ******************************************
w1 = [2.400, 3.000, 3.300, 3.600, 3.9000, 4.200]
w2 = [2.412, 3.009, 3.308, 3.607, 3.9065, 4.206]    ; wavelength that is 36 um away
R = w1/(w2-w1)

slit_image = [36., 36., 36., 36., 36., 36.]   ; slit image width in microns
spot = [14., 14., 15., 20., 24., 24.]             ; median spot size in microns
diffract = w1*12.   ; diffraction spot size in microns

blur = sqrt(slit_image^2 + spot^2 + diffract^2)
degrade = blur/slit_image

oplot, w1, r, psym=-1, color=1, thick=t
oplot, w1, r/degrade, psym=-1, color=1, linestyle=2, thick=t

print, ''
print, 'Minimum TIR resolution:', min(r),  '       Mean MIR resolution:', mean(r)
print, 'Minimum TIR resolution:', min(r/degrade),  '       Mean MIR resolution:', mean(r/degrade)

print, "MIR"
print, "Wavelength", w1
print, "Resolution", r
print, "Degraded  ", r/degrade

; ******************************************
;        Vis   230825 f/10 2 pixels/slice  
; ******************************************

w1 = [0.40000, 0.5000, 0.6000, 0.7000, 0.800, 0.8500]
w2 = [0.40070, 0.5014, 0.6023, 0.7033, 0.8045, 0.855]        ; wavelength that is a slit width (30 um)  away,
R = w1/(w2-w1)

slit_image = [30., 30., 30., 30., 30., 30.]   ; slit image width in microns
spot = [22., 15., 16., 15., 16., 16.]             ; median spot size in microns
diffract = w1*10.2   ; diffraction spot size in microns

blur = sqrt(slit_image^2 + spot^2 + diffract^2)
degrade = blur/slit_image

oplot, w1, r, psym=-1, color=7, thick=t
oplot, w1, r/degrade, psym=-1, color=7, linestyle=2, thick=t

print, ''
print, 'Minimum VIS resolution:', min(r),  '       Mean VIS resolution:', mean(r)
print, 'Minimum VIS resolution:', min(r/degrade),  '       Mean VIS resolution:', mean(r/degrade)

print, "VIS"
print, "Wavelength", w1
print, "Resolution", r
print, "Degraded  ", r/degrade

; Begin other printing things?

xx = [0.35, 5]
yy = [150, 150]
PlotS, xx, yy, linestyle=1, color=10, thick=t

xx = [0.4, 0.6]
yy = [700, 700]
PlotS, xx, yy, color=10, thick=t

xx = [0.4, 0.6]
yy = [650, 650]
PlotS, xx, yy, linestyle=2, color=10, thick=t

XYOutS, 0.3, 0.89, '0.2" slit image only', /normal, charthick=t
XYOutS, 0.3, 0.83, 'Degraded by spot size and diffraction', /normal, charthick=t

h = 0.055
XYOutS, 0.14, h, '0.4', /normal, charthick=t
XYOutS, 0.21, h, '0.5', /normal, charthick=t
XYOutS, 0.27, h, '0.6', /normal, charthick=t
XYOutS, 0.36, h, '0.8', /normal, charthick=t
XYOutS, 0.67, h, '2', /normal, charthick=t
XYOutS, 0.80, h, '3', /normal, charthick=t
XYOutS, 0.89, h, '4', /normal, charthick=t

; device, /close  ; closes device
; set_plot, 'x'   ; makes it so that the plot goes to the screen again
END
