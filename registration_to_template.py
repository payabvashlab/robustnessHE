                #baseline
                image1 = nib.load(self.folder + "/CT/" + filename1 + ".nii.gz")
                image1_gt = nib.load(self.folder + "/gt_masks/" + filename1 + "-label_bleeding.nii.gz")
                image1_gt_data = image1_gt.get_fdata()
                image1_gt = nib.Nifti1Image(image1_gt_data, image1.affine, image1.header)

                image1, _ = extract.skullstrip_2outputs(image1)
                original_image1 = convert.nii2ants(image1)
                image1, transforms1 = register.rigid(template, original_image1)
                image1, ants_params = convert.ants2np(image1)
                nii1_gt = convert.nii2ants(image1_gt)
                nii1_gt = ants.apply_transforms(template, nii1_gt, transforms1)
                mask1_gt, ants_params_seg = convert.ants2np(nii1_gt)
                mask1_gt[mask1_gt >= 0.5] = 1
                mask1_gt[mask1_gt < 0.5] = 0
                # follow-up

                image2 = nib.load(self.folder + "/CT/" + filename2 + ".nii.gz")
                image2_gt = nib.load(self.folder + "/gt_masks/" + filename2 + "-label_bleeding.nii.gz")
                image2_gt_data = image2_gt.get_fdata()
                image2_gt = nib.Nifti1Image(image2_gt_data, image2.affine, image2.header)

                image2, _ = extract.skullstrip_2outputs(image2)
                original_image2 = convert.nii2ants(image2)
                image2, transforms2 = register.rigid(template, original_image2)
                image2, ants_params = convert.ants2np(image2)
                nii2_gt = convert.nii2ants(image2_gt)
                nii2_gt = ants.apply_transforms(template, nii2_gt, transforms2)
                mask2_gt, ants_params_seg = convert.ants2np(nii2_gt)
                mask2_gt[mask2_gt >= 0.5] = 1
                mask2_gt[mask2_gt < 0.5] = 0

                normalize = NormalizeIntensity()
                scale = ScaleIntensity()
                image1 = normalize(image1[:, :, :, :, 0])
                image1 = scale(image1)
                image2 = normalize(image2[:, :, :, :, 0])
                image2 = scale(image2)
